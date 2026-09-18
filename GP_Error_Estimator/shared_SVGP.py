import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch.distributions import MultivariateNormal
from utils import *
from GP_Error_Estimator.central import CoreGPModel

import torch.nn.functional as F
import gpytorch
import torch
import torch.nn as nn
from utils import psd_safe_cholesky

from utils import psd_safe_cholesky



# ============================================================
# Gaussian likelihood for regression
# ============================================================

class GaussianLikelihood(gpytorch.Module):
    """
    Gaussian observation model:

        y = f(x) + eps
        eps ~ N(0, noise_variance)

    noise_variance is shared across all tokens and output dims.
    """

    def __init__(self, init_noise=0.0):
        super().__init__()

        # raw parameter -> positive noise through softplus
        raw_init = torch.log(
            torch.expm1(torch.tensor(init_noise))
        )

        self.raw_noise = nn.Parameter(raw_init)

    @property
    def noise(self):
        return F.softplus(self.raw_noise) + 1e-6


# ============================================================
# Helper
# ============================================================

def _triangular_inverse(A):
    """
    Inverse of a lower-triangular matrix.
    """
    I = torch.eye(
        A.size(-1),
        dtype=A.dtype,
        device=A.device
    )

    return torch.linalg.solve_triangular(
        A,
        I,
        upper=False
    )


# ============================================================
# Shared Variational ELBO
# ============================================================

class VariationalELBO(gpytorch.Module):

    def __init__(
        self,
        model,#kernel_function,
        num_data,
        num_inducing_points,
        total_tokens=96,
        output_dim=96,
        lr=0.1,
        dtype=torch.float64
    ):
        super().__init__()

        self.model = model

        self.num_data = num_data
        self.num_inducing_points = num_inducing_points

        self.total_tokens = total_tokens
        self.output_dim = output_dim

        self.lr = lr
        self.dtype = dtype

        # =====================================================
        # Variational parameters
        #
        # eta[t, d, :]
        #   t = token
        #   d = SVD/output dimension
        #
        # H[t, :, :]
        #   covariance natural parameter for token t
        #
        # H is shared across the 96 output dimensions of a token.
        # =====================================================

        self.register_buffer(
            "eta",
            torch.zeros(
                total_tokens,
                output_dim,
                num_inducing_points,
                dtype=dtype
            )
        )

        self.register_buffer(
            "H",
            -0.5 * torch.eye(
                num_inducing_points,
                dtype=dtype
            ).unsqueeze(0).repeat(
                total_tokens,
                1,
                1
            )
        )

        self.ctx = None


    # ========================================================
    # Convert natural parameters -> mu, Sigma
    # ========================================================

    def NaturalToMuSigma(self, token_idx=None):

        if token_idx is None:
            token_idx = torch.arange(
                self.total_tokens,
                device=self.eta.device
            )

        eta = self.eta[token_idx]
        H = self.H[token_idx]

        # eta:
        # [B, D, M]

        # H:
        # [B, M, M]

        # -2H = Sigma^{-1}
        precision = -2.0 * H

        L_inv = psd_safe_cholesky(precision)

        # Sigma = precision^{-1}
        L = _triangular_inverse(L_inv)

        Sigma = L.transpose(-1, -2) @ L
        # [B, M, M]

        # mu = Sigma @ eta
        mu = torch.matmul(
            Sigma.unsqueeze(1),
            eta.unsqueeze(-1)
        ).squeeze(-1)
        # [B, D, M]

        return mu, Sigma


    # ========================================================
    # ELBO
    # ========================================================

    def forward(self, K, Y, batch_idx):
        """
        Parameters
        ----------
        K:
            Full covariance matrix of [Z ; X]

            shape:
                [M + N, M + N]

        Y:
            Regression targets for current token batch

            shape:
                [N, B, D]

            N = number of samples
            B = number of tokens in current batch
            D = 96

        batch_idx:
            global token indices, e.g.

                [0,...,23]
                [24,...,47]
                [48,...,71]
                [72,...,95]

        Returns
        -------
        ELBO scalar
        """

        M = self.num_inducing_points

        # ----------------------------------------------------
        # Current token variational parameters
        # ----------------------------------------------------

        mu, Sigma = self.NaturalToMuSigma(batch_idx)

        # mu:
        # [B, D, M]

        # Sigma:
        # [B, M, M]

        # ----------------------------------------------------
        # Extract K blocks
        # ----------------------------------------------------

        Kmm = K[:M, :M]

        Knm = K[:M, M:].t()
        # [N, M]

        Knn = K[M:, M:]
        # [N, N]

        # ----------------------------------------------------
        # Cholesky of Kmm
        # ----------------------------------------------------

        L = psd_safe_cholesky(Kmm)

        I = torch.eye(
            M,
            dtype=K.dtype,
            device=K.device
        )

        Kmm_inv = torch.cholesky_solve(
            I,
            L
        )
        # [M, M]

        # ----------------------------------------------------
        # kappa = Knm Kmm^{-1}
        # ----------------------------------------------------

        kappa = torch.cholesky_solve(
            Knm.t(),
            L
        ).t()
        # [N, M]

        # ----------------------------------------------------
        # Conditional variance:
        #
        # K_tilde =
        # diag(Knn - Knm Kmm^{-1} Knm^T)
        # ----------------------------------------------------

        K_tilde = torch.diagonal(
            Knn - Knm @ kappa.t()
        )
        # [N]

        # ----------------------------------------------------
        # Re-arrange Y
        #
        # [N, B, D]
        #     ->
        # [B, D, N]
        # ----------------------------------------------------

        Y_bdn = Y.permute(1, 2, 0)

        B = Y.shape[1]
        D = Y.shape[2]
        N_batch = Y.shape[0]

        # ----------------------------------------------------
        # Predictive latent mean
        #
        # mu_f = kappa @ mu
        # ----------------------------------------------------

        mu_f = torch.einsum(
            "nm,bdm->bdn",
            kappa,
            mu
        )
        # [B, D, N]

        # ----------------------------------------------------
        # Predictive latent variance from q(u)
        #
        # diag(kappa Sigma kappa^T)
        # ----------------------------------------------------

        q_var = torch.einsum(
            "nm,bmk,nk->bn",
            kappa,
            Sigma,
            kappa
        )
        # [B, N]

        # total latent variance
        latent_var = (
            K_tilde.unsqueeze(0)
            +
            q_var
        )
        # [B, N]

        # ----------------------------------------------------
        # Gaussian likelihood
        # ----------------------------------------------------

        noise = self.model_likelihood_noise
        noise2 = noise

        # Expected squared error:
        #
        # E[(Y-f)^2]
        #
        # = Y^2 - 2Y E[f] + E[f^2]
        #
        # E[f^2] = mu_f^2 + var_f
        # ----------------------------------------------------

        sq_error = (
            Y_bdn.pow(2)
            - 2.0 * Y_bdn * mu_f
            + mu_f.pow(2)
            + latent_var.unsqueeze(1)
        )

        expected_log_likelihood = (
            -0.5 / noise2 * sq_error.sum()
            -0.5
            * N_batch
            * B
            * D
            * torch.log(
                2.0 * torch.pi * noise2
            )
        )

        # ----------------------------------------------------
        # If using minibatches of DATA samples:
        #
        # scale likelihood to full dataset
        # ----------------------------------------------------

        data_scale = self.num_data / N_batch

        expected_log_likelihood = (
            data_scale * expected_log_likelihood
        )

        # ----------------------------------------------------
        # KL[q(u) || p(u)]
        #
        # p(u) = N(0, Kmm)
        #
        # q(u) = N(mu, Sigma)
        # ----------------------------------------------------

        # log |Kmm|
        sign_K, logdet_Kmm = torch.linalg.slogdet(Kmm)

        if torch.any(sign_K <= 0):
            raise RuntimeError(
                "Kmm is not positive definite."
            )

        # log |Sigma|
        sign_S, logdet_Sigma = torch.linalg.slogdet(Sigma)

        if torch.any(sign_S <= 0):
            raise RuntimeError(
                "Variational Sigma is not positive definite."
            )

        # trace(Kmm^{-1} Sigma)
        trace_term = torch.einsum(
            "ij,bij->b",
            Kmm_inv,
            Sigma
        )
        # [B]

        # mu^T Kmm^{-1} mu
        quad_term = torch.einsum(
            "bdm,mn,bdn->bd",
            mu,
            Kmm_inv,
            mu
        )
        # [B, D]

        kl_per_output = 0.5 * (
            trace_term.unsqueeze(-1)
            + quad_term
            - M
            + logdet_Kmm
            - logdet_Sigma.unsqueeze(-1)
        )

        KL = kl_per_output.sum()

        # ----------------------------------------------------
        # ELBO
        # ----------------------------------------------------

        ELBO = expected_log_likelihood - KL

        # save information for natural update
        self.ctx = {
            "Kmm": Kmm.detach(),
            "Kmm_inv": Kmm_inv.detach(),
            "kappa": kappa.detach(),
            "Y": Y.detach(),
            "batch_idx": batch_idx.detach(),
            "N_batch": N_batch,
        }

        return ELBO


    # ========================================================
    # Natural parameter update
    # ========================================================

    @torch.no_grad()
    def update(self):

        if self.ctx is None:
            raise RuntimeError(
                "Call forward() before update()."
            )

        Kmm = self.ctx["Kmm"]
        Kmm_inv = self.ctx["Kmm_inv"]
        kappa = self.ctx["kappa"]
        Y = self.ctx["Y"]
        batch_idx = self.ctx["batch_idx"]
        N_batch = self.ctx["N_batch"]

        noise = self.model_likelihood_noise
        noise2 = noise

        data_scale = self.num_data / N_batch

        # ----------------------------------------------------
        # Y:
        # [N, B, D]
        #
        # -> [B, D, N]
        # ----------------------------------------------------

        Y_bdn = Y.permute(1, 2, 0)

        # ----------------------------------------------------
        # Optimal natural mean:
        #
        # eta* =
        # (N / noise2) Kappa^T Y
        # ----------------------------------------------------

        eta_target = (
            data_scale / noise2
            * torch.einsum(
                "nm,bdn->bdm",
                kappa,
                Y_bdn
            )
        )
        # [B, D, M]

        # ----------------------------------------------------
        # Optimal H:
        #
        # H* =
        # -1/2 [
        #       Kmm^{-1}
        #       +
        #       N/noise2 * Kappa^T Kappa
        #      ]
        #
        # Same H for all 96 output dimensions
        # of a given token.
        # ----------------------------------------------------

        kappa_t_kappa = kappa.t() @ kappa
        # [M, M]

        H_target = -0.5 * (
            Kmm_inv
            +
            data_scale / noise2
            * kappa_t_kappa
        )
        # [M, M]

        # ----------------------------------------------------
        # Natural-gradient style interpolation
        # ----------------------------------------------------

        self.eta[batch_idx] += (
            self.lr
            * (
                eta_target
                - self.eta[batch_idx]
            )
        )

        self.H[batch_idx] += (
            self.lr
            * (
                H_target.unsqueeze(0)
                - self.H[batch_idx]
            )
        )


    # ========================================================
    # Noise property
    # ========================================================

    @property
    def model_likelihood_noise(self):
        return self._noise


# ============================================================
# Main Shared SVGP model
# ============================================================

class Core_Model(gpytorch.Module):

    def __init__(
        self,
        kernel_func,
        dtype=torch.float64,
        num_inducing_points=10,
        num_data=100,
        natural_lr=0.1,
        total_tokens=96,
        tokens_per_batch=24,
        output_dim=96,
        init_noise=1e-2,
    ):

        super().__init__()

        self.num_inducing_points = num_inducing_points
        self.kernel_func = kernel_func

        self.dtype = dtype

        self.total_tokens = total_tokens
        self.tokens_per_batch = tokens_per_batch
        self.output_dim = output_dim

        # ----------------------------------------------------
        # ONE shared GP / kernel
        # ----------------------------------------------------

        self.model = CoreGPModel(
            kernel_func,
            jitter_val=1e-2
        )

        # ----------------------------------------------------
        # Shared Gaussian observation noise
        # ----------------------------------------------------

        raw_noise = torch.log(
            torch.expm1(
                torch.tensor(
                    init_noise,
                    dtype=dtype
                )
            )
        )

        self.raw_noise = nn.Parameter(
            raw_noise
        )

        # ----------------------------------------------------
        # Shared variational ELBO
        # ----------------------------------------------------

        self.ELBO = VariationalELBO(
            self.model,
            num_data=num_data,
            num_inducing_points=num_inducing_points,
            total_tokens=total_tokens,
            output_dim=output_dim,
            lr=natural_lr,
            dtype=dtype
        )

        # connect likelihood noise to ELBO
        self.ELBO._noise = self.noise


    @property
    def noise(self):
        return F.softplus(self.raw_noise) + 1e-6


    # ========================================================
    # Forward / ELBO
    # ========================================================

    def forward_mll(
        self,
        Points,
        Y,
        batch_idx,
        to_print=False
    ):
        """
        Points:
            [M + N, input_dim]

        Y:
            [N, 24, 96]

        batch_idx:
            e.g.
            tensor([0,...,23])
        """

        _, K = self.model(Points)

        # Update current likelihood noise reference
        self.ELBO._noise = self.noise

        mll = self.ELBO(
            K,
            Y,
            batch_idx
        )

        
        print("Y shape:", Y.shape)
        print("K shape:", K.shape)
        print("noise:", self.noise.item())
        print("ELBO:", mll.item())

        return mll


    # ========================================================
    # Predictive posterior
    # ========================================================

    @torch.no_grad()
    def predictive_posterior(
        self,
        Points,
        batch_idx,
        jitter_Kmm=False,
        include_noise=True
    ):
        """
        Predictive regression.

        Returns:

            mean:
                [B, N, 96]

            variance:
                [B, N, 96]
        """

        _, K = self.model(Points)

        M = self.num_inducing_points

        I = torch.eye(
            M,
            dtype=K.dtype,
            device=K.device
        )

        Kmm = K[:M, :M]

        if jitter_Kmm:
            Kmm = Kmm + 0.03 * I

        Knm = K[:M, M:].t()
        # [N, M]

        Knn = K[M:, M:]

        L = psd_safe_cholesky(Kmm)

        Kmm_inv = torch.cholesky_solve(
            I,
            L
        )

        # kappa = Knm Kmm^{-1}
        kappa = torch.cholesky_solve(
            Knm.t(),
            L
        ).t()

        # ----------------------------------------------------
        # Current token posterior
        # ----------------------------------------------------

        mu, Sigma = self.ELBO.NaturalToMuSigma(
            batch_idx
        )

        # mu:
        # [B, D, M]

        # Sigma:
        # [B, M, M]

        # ----------------------------------------------------
        # Predictive mean
        # ----------------------------------------------------

        mu_s = torch.einsum(
            "nm,bdm->bnd",
            kappa,
            mu
        )

        # [B, N, D]

        # ----------------------------------------------------
        # Conditional GP variance
        # ----------------------------------------------------

        K_tilde = torch.diagonal(
            Knn
            - Knm @ kappa.t()
        )
        # [N]

        # variance from q(u)
        q_var = torch.einsum(
            "nm,bmk,nk->bn",
            kappa,
            Sigma,
            kappa
        )
        # [B, N]

        Sigma_s = (
            K_tilde.unsqueeze(0).unsqueeze(-1)
            +
            q_var.unsqueeze(-1)
        )

        # [B, N, 1]

        Sigma_s = Sigma_s.expand(
            -1,
            -1,
            self.output_dim
        )

        # ----------------------------------------------------
        # Add observation noise for y
        # ----------------------------------------------------

        if include_noise:
            Sigma_s = Sigma_s + self.noise

        return mu_s, Sigma_s

import torch
import torch.nn as nn
import gpytorch

from utils import psd_safe_cholesky
from GP_Error_Estimator.central import CoreGPModel


# ============================================================
# Low-Rank utilities
# ============================================================

def low_rank_decomposition(K, rank, jitter=1e-6):
    """
    Low-rank + diagonal approximation:

        K ≈ Q Λ Q^T + jitter * I

    where:
        Q      : [M, r]
        Λ      : [r]

    Parameters
    ----------
    K : Tensor
        [M, M] PSD kernel matrix.

    rank : int
        Number of retained eigen-components.

    jitter : float
        Residual diagonal term used to keep the approximation
        strictly positive definite.

    Returns
    -------
    Q : [M, r]
        Leading eigenvectors.

    eigvals_r : [r]
        Leading eigenvalues.
    """

    # Force symmetry
    K = 0.5 * (K + K.transpose(-1, -2))

    M = K.size(-1)
    rank = min(rank, M)

    # Full eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(K)

    # Keep largest eigenvalues
    eigvals_r = eigvals[-rank:]
    Q = eigvecs[:, -rank:]

    # Numerical safety
    eigvals_r = eigvals_r.clamp_min(jitter)

    return Q, eigvals_r


def low_rank_inverse_apply(A, Q, eigvals, jitter):
    """
    Compute approximately:

        K^{-1} A

    where

        K ≈ Q Λ Q^T + jitter I

    using Woodbury identity.

    Supports:
        A = [M, P]
        A = [B, M, P]

    Returns
    -------
    K_inv_A
    """

    inv_jitter = 1.0 / jitter

    # Λ / (δ + Λ)
    coeff = eigvals / (jitter + eigvals)

    # Q^T A
    projected = Q.transpose(-1, -2) @ A

    # Q diag(coeff) Q^T A
    correction = Q @ (
        coeff.unsqueeze(-1) * projected
    )

    result = (
        inv_jitter * A
        -
        inv_jitter * correction
    )

    return result


def low_rank_logdet(M, Q, eigvals, jitter):
    """
    log |K|

    for

        K ≈ Q Λ Q^T + jitter I

    Using:

        log|K|
        =
        (M-r) log(jitter)
        +
        sum log(jitter + lambda_i)
    """

    rank = eigvals.numel()

    residual_dim = M - rank

    logdet = (
        residual_dim * torch.log(
            torch.as_tensor(
                jitter,
                dtype=eigvals.dtype,
                device=eigvals.device
            )
        )
        +
        torch.log(jitter + eigvals).sum()
    )

    return logdet


# ============================================================
# Variational ELBO
# ============================================================

class VariationalELBO(gpytorch.Module):

    def __init__(
        self,
        model,
        num_data,
        num_inducing_points,
        lr=0.1,
        dtype=torch.float64,
        low_rank_rank=4,
        low_rank_jitter=1e-5
    ):
        super().__init__()

        self.model = model

        # ----------------------------------------------------
        # Training information
        # ----------------------------------------------------

        self.lr = lr
        self.num_data = num_data

        self.num_inducing_points = num_inducing_points

        # ----------------------------------------------------
        # Low-rank parameters
        # ----------------------------------------------------

        self.low_rank_rank = 22#low_rank_rank
        self.low_rank_jitter = 1e-1#low_rank_jitter

        # ----------------------------------------------------
        # Dimensions
        # ----------------------------------------------------

        self.total_tokens = 1
        self.output_dim = 294912

        self.D = 294912
        self.N = 44
        self.B = 1
        self.M = num_inducing_points

        # ----------------------------------------------------
        # Natural parameters
        #
        # eta:
        # [tokens, D, M]
        #
        # H:
        # [tokens, M, M]
        # ----------------------------------------------------
        natural_dtype = torch.float64

        self.register_buffer(
                       "eta",
                       torch.zeros(
                       self.total_tokens,
                       self.output_dim,
                       self.num_inducing_points,
                       dtype=natural_dtype))

        self.register_buffer(
                          "H", -0.5 * torch.eye(
                                        self.num_inducing_points,
                                        dtype=natural_dtype
                                         ).unsqueeze(0).repeat(
                                        self.total_tokens,1,1))
        
        # ----------------------------------------------------
        # Observation noise
        # ----------------------------------------------------

        self.register_buffer(
            "noise",
            torch.tensor(
                5e-5,
                dtype=dtype
            )
        )

    # ========================================================
    # Forward ELBO
    # ========================================================

    def forward(self, K, Y, batch_idx):

        M = self.num_inducing_points

        # ----------------------------------------------------
        # 1) Make sure Y has [N, B, D]
        # ----------------------------------------------------

        if Y.ndim == 2:
            Y = Y.unsqueeze(1)

        # ----------------------------------------------------
        # 2) Variational posterior
        # ----------------------------------------------------

        mu, Sigma = self.NaturalToMuSigma(batch_idx)

        # Everything follows kernel dtype
        mu = mu.to(K.dtype)
        Sigma = Sigma.to(K.dtype)
        Y = Y.to(K.dtype)

        # ----------------------------------------------------
        # 3) Split full kernel
        #
        # K = [Kmm  Kmn
        #      Knm  Knn]
        # ----------------------------------------------------

        Kmm = K[:M, :M]

        Knm = K[:M, M:].transpose(0, 1)

        Knn = K[M:, M:]

        # ----------------------------------------------------
        # 4) Low-rank approximation of Kmm
        # ----------------------------------------------------

        rank = min(
            self.low_rank_rank,
            M
        )


        Q, eigvals = low_rank_decomposition(
                         Kmm,
                         rank=rank,
                         jitter=self.low_rank_jitter
          )

        Kmm_inv_Kmn = low_rank_inverse_apply(
                  Knm.transpose(0, 1),
                  Q,
                  eigvals,
                  self.low_rank_jitter
                  )

        kappa = Kmm_inv_Kmn.transpose(0, 1)

        # ----------------------------------------------------
        # 5) Kmm^{-1} approximation
        #
        # Kmm ≈ QΛQ^T + δI
        # ----------------------------------------------------

        Kmm_inv_mu = low_rank_inverse_apply(
                      mu.transpose(-1, -2),
                      Q,
                      eigvals,
                      self.low_rank_jitter
                      ).transpose(-1, -2)

        # ----------------------------------------------------
        # 6) kappa
        #
        # kappa = Knm Kmm^{-1}
        #
        # [N, M]
        # ----------------------------------------------------

        kappa = Kmm_inv_Kmn.transpose(0, 1)

        # ----------------------------------------------------
        # 7) Predictive mean
        #
        # mu_f = kappa @ mu
        # ----------------------------------------------------

        kappaMu = torch.einsum(
            "nm,bdm->bdn",
            kappa,
            mu
        )

        # [B, D, N] -> [B, N, D]

        mu_f = kappaMu.permute(
            0,
            2,
            1
        )

        # ----------------------------------------------------
        # 8) Conditional GP variance
        #
        # diag(
        #   Knn - Knm Kmm^{-1} Kmn
        # )
        # ----------------------------------------------------

        K_tilde = torch.diagonal(
            Knn
            -
            Knm @ kappa.transpose(0, 1)
        )

        # Numerical protection
        K_tilde = K_tilde.clamp_min(0.0)

        # ----------------------------------------------------
        # 9) Variational uncertainty
        #
        # diag(
        #   kappa Sigma kappa^T
        # )
        # ----------------------------------------------------

        kappaSigmakappa = torch.einsum(
            "nm,bmk,nk->bn",
            kappa,
            Sigma,
            kappa
        )

        # ----------------------------------------------------
        # 10) Total predictive variance
        # ----------------------------------------------------

        predictive_var = (
            K_tilde.unsqueeze(0)
            +
            kappaSigmakappa
        )

        # [B, N] -> [B, N, D]

        predictive_var = (
            predictive_var
            .unsqueeze(-1)
            .expand(
                self.B,
                self.N,
                self.D
            )
        )

        # ----------------------------------------------------
        # 11) Expected squared error
        # ----------------------------------------------------

        Y_bnd = Y.permute(
            1,
            0,
            2
        )

        expected_sq_error = (
            (Y_bnd - mu_f).pow(2)
            +
            predictive_var
        )

        # ----------------------------------------------------
        # 12) Expected log likelihood
        # ----------------------------------------------------

        expected_log_likelihood = (
            -0.5
            *
            (
                expected_sq_error / self.noise
                +
                torch.log(
                    2.0
                    * torch.pi
                    * self.noise
                )
            ).sum()
        )

        # ====================================================
        # 13) KL[q(u) || p(u)]
        # ====================================================

        # ----------------------------------------------------
        # log |Kmm|
        # ----------------------------------------------------

        logdet_Kmm = low_rank_logdet(
            M=M,
            Q=Q,
            eigvals=eigvals,
            jitter=self.low_rank_jitter
        )

        # ----------------------------------------------------
        # log |Sigma|
        # ----------------------------------------------------

        sign_S, logdet_Sigma = torch.linalg.slogdet(
            Sigma
        )

        if torch.any(sign_S <= 0):
            raise RuntimeError(
                "Variational Sigma is not positive definite."
            )

        # ----------------------------------------------------
        # Kmm^{-1} Sigma
        # ----------------------------------------------------

        Kmm_inv_Sigma = low_rank_inverse_apply(
            Sigma,
            Q,
            eigvals,
            self.low_rank_jitter
        )

        # ----------------------------------------------------
        # trace(Kmm^{-1} Sigma)
        # ----------------------------------------------------

        trace_term = torch.diagonal(
            Kmm_inv_Sigma,
            dim1=-2,
            dim2=-1
        ).sum(dim=-1)

        # ----------------------------------------------------
        # Kmm^{-1} mu
        # ----------------------------------------------------

        # mu:
        # [B, D, M]
        #
        # We transform it to:
        # [B, D, M]

        Kmm_inv_mu = low_rank_inverse_apply(
                     mu.transpose(-1, -2),
                     Q,
                     eigvals,
                     self.low_rank_jitter
                    ).transpose(-1, -2)

        # ----------------------------------------------------
        # mu^T Kmm^{-1} mu
        # ----------------------------------------------------

        quad_term = (
            mu
            *
            Kmm_inv_mu
        ).sum(dim=-1)

        # [B, D]

        # ----------------------------------------------------
        # KL per output
        # ----------------------------------------------------

        KL_per_output = 0.5 * (
            trace_term.unsqueeze(-1)
            +
            quad_term
            -
            M
            +
            logdet_Kmm
            -
            logdet_Sigma.unsqueeze(-1)
        )

        # ----------------------------------------------------
        # Total KL
        # ----------------------------------------------------

        KL = KL_per_output.sum()

        # ====================================================
        # 14) ELBO
        # ====================================================

        ELBO = (
            expected_log_likelihood
            -
            KL
        )

        # ----------------------------------------------------
        # Save context
        # ----------------------------------------------------

        self.ctx = {
            "Kmm": Kmm.detach(),
            "Knm": Knm.detach(),
            "kappa": kappa.detach(),

            "Q": Q.detach(),
            "eigvals": eigvals.detach(),

            "Y": Y.detach(),
            "batch_idx": batch_idx.detach(),

            "predictive_var":
                predictive_var.detach()
        }

        return ELBO

    # ========================================================
    # Natural -> Mean / Covariance
    # ========================================================

    def NaturalToMuSigma(self, batch_idx):

        eta = self.eta[batch_idx]

        H = self.H[batch_idx]
        

        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        precision = -2.0 * H

        precision = 0.5 * (precision + precision.transpose(-1, -2))

        eigvals_precision = torch.linalg.eigvalsh(precision)

        #print("precision min eig:", eigvals_precision.min().item())

        #print("precision max eig:", eigvals_precision.max().item())

        L_precision = psd_safe_cholesky(precision)

        M = H.shape[-1]

        I = torch.eye(
            M,
            dtype=H.dtype,
            device=H.device
        ).expand(
            H.shape[0],
            -1,
            -1
        )

        # ----------------------------------------------------
        # Sigma = Precision^{-1}
        # ----------------------------------------------------

        Sigma = torch.cholesky_solve(
            I,
            L_precision
        )

        # ----------------------------------------------------
        # mu = Sigma @ eta
        # ----------------------------------------------------

        mu = torch.einsum(
            "bij,bdj->bdi",
            Sigma,
            eta
        )

        return mu, Sigma

    # ========================================================
    # Natural parameter update
    # ========================================================

    def update(self):
        
        Kmm = self.ctx["Kmm"]

        kappa = self.ctx["kappa"]

        Y = self.ctx["Y"]

        batch_idx = self.ctx["batch_idx"]

        with torch.no_grad():

            # ------------------------------------------------
            # Shapes
            # ------------------------------------------------

            N_batch = Y.shape[0]

            B = Y.shape[1]

            D = Y.shape[2]

            M = self.num_inducing_points

            # ------------------------------------------------
            # Minibatch scaling
            # ------------------------------------------------

            data_scale = (
                self.num_data
                /
                N_batch
            )

            # ------------------------------------------------
            # Recompute low-rank factors
            # ------------------------------------------------

            rank = min(
                self.low_rank_rank,
                M
            )

            Q, eigvals = low_rank_decomposition(
                Kmm,
                rank=rank,
                jitter=self.low_rank_jitter
            )

            # ------------------------------------------------
            # Kmm^{-1} approximation
            # ------------------------------------------------

            I = torch.eye(
                M,
                dtype=Kmm.dtype,
                device=Kmm.device
            )

            Kmm_inv = low_rank_inverse_apply(
                I,
                Q,
                eigvals,
                self.low_rank_jitter)
            
            Kmm = 0.5 * (Kmm + Kmm.transpose(-1, -2))

            jitter = 1e-5
            I = torch.eye(M, dtype=Kmm.dtype, device=Kmm.device)

            Kmm_reg = Kmm + jitter * I

            L = torch.linalg.cholesky(Kmm_reg)

            
            #print("\n===== KMM_INV DEBUG =====")

            Kinv_sym = 0.5 * (Kmm_inv + Kmm_inv.transpose(-1, -2))

            eig_Kinv = torch.linalg.eigvalsh(Kinv_sym)

            #print("Kmm_inv min:", Kmm_inv.min().item())
            #print("Kmm_inv max:", Kmm_inv.max().item())
            #print("Kmm_inv min eig:", eig_Kinv.min().item())
            #print("Kmm_inv max eig:", eig_Kinv.max().item())

            #print("=========================\n")
            # ------------------------------------------------
            # Y:
            #
            # [N, B, D] -> [B, D, N]
            # ------------------------------------------------

            Y_bdn = Y.permute(
                1,
                2,
                0
            )

            # =================================================
            # eta target
            #
            # eta* =
            #
            # N/noise * Kappa^T Y
            #
            # =================================================

            eta_target = (
                data_scale
                /
                self.noise.to(Kmm.dtype)
                *
                torch.einsum(
                    "nm,bdn->bdm",
                    kappa,
                    Y_bdn
                )
            )

            # =================================================
            # H target
            #
            # H* =
            #
            # -1/2 [
            #
            #   Kmm^{-1}
            #
            #   +
            #
            #   N/noise Kappa^T Kappa
            #
            # ]
            #
            # =================================================

            # ============================================================
            # Stable H_target computation in float64
            # ============================================================

            kappa64 = kappa.to(torch.float64)
            Kmm_inv64 = Kmm_inv.to(torch.float64)

            scale64 = (data_scale.to(torch.float64)/self.noise.to(torch.float64))

            # Gram matrix: kappa^T kappa
            kappa_t_kappa64 = (kappa64.transpose(0, 1) @ kappa64)

            # Enforce exact symmetry numerically
            kappa_t_kappa64 = 0.5 * (kappa_t_kappa64 + kappa_t_kappa64.transpose(-1, -2))

            # Inspect Gram matrix
            eig_KtK64 = torch.linalg.eigvalsh(kappa_t_kappa64)

            #print("\n===== FLOAT64 KtK DEBUG =====")
            #print("KtK min eig:", eig_KtK64.min().item())
            #print("KtK max eig:", eig_KtK64.max().item())
            #print("==============================\n")

            # ============================================================
            # A = Kmm^{-1} + (N/noise) Kappa^T Kappa
            # ============================================================

            A_target64 = (Kmm_inv64 + scale64 * kappa_t_kappa64)

            # Enforce symmetry
            A_target64 = 0.5 * (A_target64 + A_target64.transpose(-1, -2))

            eig_A64 = torch.linalg.eigvalsh(A_target64)

            #print("\n===== FLOAT64 A_TARGET DEBUG =====")
            #print("A_target min eig:", eig_A64.min().item())
            #print("A_target max eig:", eig_A64.max().item())
            #print("==================================\n")

            # ============================================================
            # Numerical PSD protection
            # A_target should theoretically be PSD
            # ============================================================

            eig_A64, eigvec_A64 = torch.linalg.eigh(A_target64)

            eig_A64 = eig_A64.clamp_min(0.0)

            A_target64 = (eigvec_A64@ torch.diag_embed(eig_A64)@ eigvec_A64.transpose(-1, -2))

            # ============================================================
            # Natural parameter
            # ============================================================

            H_target64 = -0.5 * A_target64

            H_target64 = 0.5 * (H_target64 + H_target64.transpose(-1, -2))

            H_target = H_target64.to(self.H.dtype)

            # =================================================
            # H TARGET DEBUG
            # =================================================

            #print("\n===== H TARGET DEBUG =====")

            H_before = self.H[batch_idx]

            H_before_sym = 0.5 * ( H_before + H_before.transpose(-1, -2))

            eig_before = torch.linalg.eigvalsh(H_before_sym)

            #print("H before min:", H_before.min().item())
            #print("H before max:", H_before.max().item())
            #print("H before min eig:", eig_before.min().item())
            #print("H before max eig:", eig_before.max().item())

            H_target_sym = 0.5 * (H_target + H_target.transpose(-1, -2))

            eig_target = torch.linalg.eigvalsh(H_target_sym)

            #print("H_target min:", H_target.min().item())
            #print("H_target max:", H_target.max().item())
            #print("H_target abs max:", H_target.abs().max().item())
            #print("H_target min eig:", eig_target.min().item())
            #print("H_target max eig:", eig_target.max().item())

            H_new = (H_before + self.lr * (H_target.unsqueeze(0).to(self.H.dtype) -H_before))

            H_new_sym = 0.5 * (H_new + H_new.transpose(-1, -2))

            eig_new = torch.linalg.eigvalsh(H_new_sym)

            #print("H after min eig:", eig_new.min().item())
            #print("H after max eig:", eig_new.max().item())

            #print("==========================\n")

            # =================================================
            # Natural gradient style update
            # =================================================

            self.eta[batch_idx] += (self.lr*(eta_target.to(self.eta.dtype)-self.eta[batch_idx]))

            self.H[batch_idx] = H_new
            '''
            # ------------------------------------------------
            # Natural gradient style update
            # ------------------------------------------------

            self.eta[batch_idx] += (
                self.lr
                *
                (
                    eta_target.to(
                        self.eta.dtype
                    )
                    -
                    self.eta[batch_idx]
                )
            )

            self.H[batch_idx] += (
                self.lr
                *
                (
                    H_target.unsqueeze(0).to(
                        self.H.dtype
                    )
                    -
                    self.H[batch_idx]
                )
            )'''


# ============================================================
# Core Model
# ============================================================

class Primary_MDL(gpytorch.Module):

    def __init__(
        self,
        kernel_func,
        dtype=torch.float32,
        num_inducing_points=22,
        num_data=44,
        natural_lr=0.9,

        # ----------------------------------------------------
        # NEW:
        # number of low-rank components
        # ----------------------------------------------------

        low_rank_rank=4,

        # ----------------------------------------------------
        # NEW:
        # residual diagonal / numerical stability
        # ----------------------------------------------------

        low_rank_jitter=1e-5
    ):

        super(Primary_MDL, self).__init__()

        self.num_inducing_points = (
            num_inducing_points
        )

        self.kernel_func = kernel_func

        self.dtype = dtype

        self.loss_fn = nn.CrossEntropyLoss()

        # ----------------------------------------------------
        # GP model
        # ----------------------------------------------------

        self.model = CoreGPModel(
            kernel_func,
            jitter_val=1e-2
        )

        # ----------------------------------------------------
        # ELBO
        # ----------------------------------------------------

        self.ELBO = VariationalELBO(
            self.model,
            num_data,
            num_inducing_points,
            natural_lr,
            dtype,
            low_rank_rank=low_rank_rank,
            low_rank_jitter=low_rank_jitter
        )

        # ----------------------------------------------------
        # Token indices
        # ----------------------------------------------------

        self.num_token = [0]

        self.token_idx = torch.tensor(
            self.num_token,
            dtype=torch.long
        )

    # ========================================================
    # Forward MLL / ELBO
    # ========================================================

    def forward_mll(
        self,
        Points,
        Y,
        batch_idx
    ):

        Y_n = Y

        _, K = self.model(Points)

        mll = self.ELBO(
            K,
            Y_n,
            self.token_idx
        )

        return mll

    # ========================================================
    # Predictive posterior
    # ========================================================

    @torch.no_grad()
    def predictive_posterior(
        self,
        Points,
        jitter_Kmm=False,
        include_noise=False
    ):

        # ----------------------------------------------------
        # Kernel matrix
        # ----------------------------------------------------

        _, K = self.model(Points)

        M = self.num_inducing_points

        # ----------------------------------------------------
        # Variational posterior
        # ----------------------------------------------------

        mu, Sigma = self.ELBO.NaturalToMuSigma(
            self.token_idx
        )

        mu = mu.to(K.dtype)
        Sigma = Sigma.to(K.dtype)

        B = mu.shape[0]
        D = mu.shape[1]

        # ----------------------------------------------------
        # Split K
        # ----------------------------------------------------

        Kmm = K[:M, :M]

        Knm = K[:M, M:].transpose(
            0,
            1
        )

        Knn = K[M:, M:]

        # ----------------------------------------------------
        # Optional additional jitter
        # ----------------------------------------------------

        if jitter_Kmm:

            I = torch.eye(
                M,
                dtype=Kmm.dtype,
                device=Kmm.device
            )

            Kmm = (
                Kmm
                +
                0.03 * I
            )

        # ----------------------------------------------------
        # Low-rank decomposition
        # ----------------------------------------------------

        rank = min(
            self.ELBO.low_rank_rank,
            M
        )

        Q, eigvals = low_rank_decomposition(
            Kmm,
            rank=rank,
            jitter=self.ELBO.low_rank_jitter
        )

        # ----------------------------------------------------
        # Approximate:
        #
        # Kmm^{-1}
        # ----------------------------------------------------

        Kmm_inv_Kmn = low_rank_inverse_apply(
            Knm.transpose(0, 1),
            Q,
            eigvals,
            self.ELBO.low_rank_jitter
        )

        # ----------------------------------------------------
        # kappa
        # ----------------------------------------------------

        kappa = (
            Kmm_inv_Kmn.transpose(
                0,
                1
            )
        )

        # ====================================================
        # Predictive mean
        # ====================================================

        mu_s = torch.einsum(
            "nm,bdm->bnd",
            kappa,
            mu
        )

        # ====================================================
        # Conditional variance
        # ====================================================

        K_tilde = torch.diagonal(
            Knn
            -
            Knm
            @
            kappa.transpose(0, 1)
        )

        K_tilde = K_tilde.clamp_min(0.0)

        # ====================================================
        # Variational variance
        # ====================================================

        kappaSigmakappa = torch.einsum(
            "nm,bmk,nk->bn",
            kappa,
            Sigma,
            kappa
        )

        # ====================================================
        # Total variance
        # ====================================================

        Sigma_s = (
            K_tilde.unsqueeze(0)
            +
            kappaSigmakappa
        )

        # ----------------------------------------------------
        # Same uncertainty across D dimensions
        # ----------------------------------------------------

        Sigma_s = (
            Sigma_s
            .unsqueeze(-1)
            .expand(
                B,
                Sigma_s.shape[1],
                D
            )
        )

        # ----------------------------------------------------
        # Optional observation noise
        # ----------------------------------------------------

        if include_noise:

            Sigma_s = (
                Sigma_s
                +
                self.ELBO.noise.to(
                    Sigma_s.dtype
                )
            )

        return mu_s, Sigma_s

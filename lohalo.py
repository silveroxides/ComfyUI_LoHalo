import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LohaloKernelScalingSampler(nn.Module):
    def __init__(self, contrast=3.38589, kernel_radius=2):
        super(LohaloKernelScalingSampler, self).__init__()
        self.contrast = contrast
        self.kernel_radius = kernel_radius
        # Pre-calculate constants for Robidoux cubic
        self.sqrt2 = math.sqrt(2.0)
        self.register_buffer('contrast_tensor', torch.tensor(contrast))

    def sigmoid_function(self, p):
        """
        Sigmoidization: Resampling through a colorspace where gamut extremes 
        are far from midtones to minimize over/undershoots.
        """
        return torch.tanh(0.5 * self.contrast * (p - 0.5))

    def inverse_sigmoid(self, q):
        """
        Inverse of the extended sigmoidal function.
        Maps the sigmoidized values back to linear light.
        """
        # Note: The C code implements a linear extension beyond 0 and 1.
        # For GPU efficiency, we approximate or stick to the core range logic
        # assuming normalized inputs [0,1].
        
        sig1 = math.tanh(0.5 * self.contrast * 0.5)
        sig0 = -sig1
        
        # Core inverse logic derived from C:
        # q = (2/contrast) * atanh( (2*sig1)*p + sig0 ) + 0.5
        # We need to map q (input p in C terms) back.
        
        # Let's follow the C logic for 'inverse_sigmoidal' which takes linear p -> curved q
        # Wait, C code naming is tricky. 
        # C: inverse_sigmoidal(p) takes linear pixel p and returns curved value.
        # C: extended_sigmoidal(q) takes curved value q and returns linear pixel.
        
        # Slope calculation for linear extension
        slope = (1.0/sig1 + sig0) * 0.25 * self.contrast
        one_over_slope = 1.0 / slope
        
        # Masking for linear extension regions
        mask_low = (q <= 0.0).float()
        mask_high = (q >= 1.0).float()
        mask_mid = 1.0 - mask_low - mask_high
        
        # Linear parts
        res_low = q * one_over_slope
        res_high = q * one_over_slope + (1.0 - one_over_slope)
        
        # Mid part (The actual inverse sigmoid)
        ssq = (2.0 * sig1) * q + sig0
        # Clamp ssq to avoid NaNs in atanh due to float precision
        ssq = torch.clamp(ssq, -0.999999, 0.999999)
        res_mid = (2.0 / self.contrast) * torch.atanh(ssq) + 0.5
        
        return res_low * mask_low + res_high * mask_high + res_mid * mask_mid

    def extended_sigmoid(self, q):
        """
        Maps curved value q back to linear pixel value p.
        """
        sig1 = math.tanh(0.5 * self.contrast * 0.5)
        slope = (1.0/sig1 - sig1) * 0.25 * self.contrast
        
        mask_low = (q <= 0.0).float()
        mask_high = (q >= 1.0).float()
        mask_mid = 1.0 - mask_low - mask_high
        
        slope_times_q = slope * q
        
        res_low = slope_times_q
        res_high = slope_times_q + (1.0 - slope)
        
        # Mid part
        # p = (0.5/sig1) * tanh(0.5*contrast*q - 0.25*contrast) + 0.5
        # Simplified: tanh(0.5*C*q + offset)
        arg = (0.5 * self.contrast) * q + (-0.25 * self.contrast)
        res_mid = (0.5 / sig1) * torch.tanh(arg) + 0.5
        
        return res_low * mask_low + res_high * mask_high + res_mid * mask_mid

    def robidoux_weight(self, r2):
        """
        Computes the Robidoux Keys Cubic weight for EWA.
        r2 is squared distance.
        """
        # Constants from C code
        # B = 1656 / (1592 + 597 * sqrt(2))
        # C_val = 15407 / (35422 + 42984 * sqrt(2))
        # The C code uses a fast polynomial approximation.
        
        # Polynomial coefficients for r < 1 (r2 < 1)
        # weight = r2 * (a3 * r + a2) + a0
        a3 = -3.0
        a2 = (45739.0 + 7164.0 * self.sqrt2) / 10319.0
        a0 = (-8926.0 - 14328.0 * self.sqrt2) / 10319.0
        
        # For 1 <= r < 2
        # weight = (r + minus_inner_root) * (r + minus_outer_root)^2
        minus_inner_root = (-103.0 - 36.0 * self.sqrt2) / (7.0 + 72.0 * self.sqrt2)
        minus_outer_root = -2.0
        
        r = torch.sqrt(r2 + 1e-8)
        
        mask_inner = (r2 < 1.0).float()
        mask_outer = ((r2 >= 1.0) & (r2 < 4.0)).float()
        
        w_inner = r2 * (a3 * r + a2) + a0
        w_outer = (r + minus_inner_root) * (r + minus_outer_root)**2
        
        # Normalize factor mentioned in C code: -398/(7+72sqrt(2))
        # However, since we normalize total weights at the end, constant scaling cancels out.
        return (w_inner * mask_inner + w_outer * mask_outer)

    def mitchell_weights(self, x):
        """
        Computes 1D Mitchell-Netravali weights for 4 points: -1, 0, 1, 2
        relative to x in [0, 1].
        Using the fast 13-flop method from Robidoux.
        """
        ax = torch.abs(x)
        
        # 7/18 * ax
        xt1 = (7.0/18.0) * ax
        xt2 = 1.0 - ax
        
        # (xt1 - 1/3) * ax * ax
        fou = (xt1 - (1.0/3.0)) * ax * ax
        
        # (1/18 - xt1) * xt2 * xt2
        one = ((1.0/18.0) - xt1) * xt2 * xt2
        
        xt3 = fou - one
        
        # Weights
        # thr = ax - fou - xt3
        thr = ax - fou - xt3
        # two = xt2 - one + xt3
        two = xt2 - one + xt3
        
        # Re-ordering to match offsets -1, 0, 1, 2
        # The C code calculates weights for positions relative to anchor.
        # uno corresponds to -1, dos to 0, tre to 1, qua to 2.
        
        # Wait, let's map C variables to standard 4-point spline indices:
        # uno -> weight for pixel at -1
        # dos -> weight for pixel at 0
        # tre -> weight for pixel at 1
        # qua -> weight for pixel at 2
        
        # In C:
        # uno = (1/18 - yt1)*yt2^2 ... wait, that's Y.
        # Let's look at the X calc:
        # one = (1/18 - xt1)*xt2^2
        # fou = (xt1 - 1/3)*ax^2
        # xt3 = fou - one
        # thr = ax - fou - xt3
        # two = xt2 - one + xt3
        
        # The C code combines these into 'uno', 'dos', 'tre', 'qua' using tensor products.
        # Here we return the 4 weights for the 1D kernel.
        # w_minus_1 = one
        # w_0 = two
        # w_1 = thr
        # w_2 = fou
        
        return one, two, thr, fou

    def get_jacobian(self, grid, height, width):
        """
        Computes the Jacobian of the grid transformation using finite differences.
        Grid is (B, H, W, 2) in pixel coordinates.
        Returns J (B, H, W, 2, 2)
        """
        # Central differences for internal pixels, forward/backward for edges
        # This is an approximation.
        
        # d(grid)/dx
        # Pad grid to handle boundaries
        grid_pad_x = F.pad(grid, (0, 0, 1, 1, 0, 0), mode='replicate')
        dx = (grid_pad_x[:, :, 2:, :] - grid_pad_x[:, :, :-2, :]) * 0.5
        
        # d(grid)/dy
        grid_pad_y = F.pad(grid, (0, 0, 0, 0, 1, 1), mode='replicate')
        dy = (grid_pad_y[:, 2:, :, :] - grid_pad_y[:, :-2, :, :]) * 0.5
        
        # J = [ dx_u  dy_u ]
        #     [ dx_v  dy_v ]
        # Where u, v are grid output coordinates (source image coords)
        # and x, y are target image coordinates.
        
        J = torch.stack([dx, dy], dim=-2) # (B, H, W, 2, 2)
        # J[..., 0, 0] is du/dx
        # J[..., 0, 1] is dv/dx
        # J[..., 1, 0] is du/dy
        # J[..., 1, 1] is dv/dy
        
        # Actually, standard Jacobian definition J_ij = d(f_i)/d(x_j)
        # f = (u, v), x = (x, y)
        # J = [ du/dx  du/dy ]
        #     [ dv/dx  dv/dy ]
        
        # Our dx calculation above gives d(grid)/dx. grid contains (u, v).
        # So dx is [du/dx, dv/dx].
        # dy is [du/dy, dv/dy].
        
        # We want J = [[du/dx, du/dy], [dv/dx, dv/dy]]
        J = torch.cat([dx.unsqueeze(-1), dy.unsqueeze(-1)], dim=-1)
        # Now J is (B, H, W, 2, 2) -> [ [du/dx, du/dy], [dv/dx, dv/dy] ]
        
        return J

    def forward(self, image, grid):
        """
        image: (B, C, H_in, W_in)
        grid: (B, H_out, W_out, 2) containing coordinates in input image space (pixels).
              NOT normalized [-1, 1].
        """
        B, C, H_in, W_in = image.shape
        B, H_out, W_out, _ = grid.shape
        
        # 1. Jacobian Analysis & Ellipse Parameters
        J = self.get_jacobian(grid, H_out, W_out)
        
        # Inverse Jacobian Jinv
        # Jinv = inv(J)
        # For 2x2 matrix [[a, b], [c, d]], inv is 1/det * [[d, -b], [-c, a]]
        det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        det = det.unsqueeze(-1).unsqueeze(-1) + 1e-8 # Avoid div zero
        
        Jinv = torch.zeros_like(J)
        Jinv[..., 0, 0] =  J[..., 1, 1]
        Jinv[..., 0, 1] = -J[..., 0, 1]
        Jinv[..., 1, 0] = -J[..., 1, 0]
        Jinv[..., 1, 1] =  J[..., 0, 0]
        Jinv = Jinv / det
        
        # Compute N = Jinv * Jinv^T
        # N11 = a^2 + b^2, N12 = ac + bd, N22 = c^2 + d^2
        a, b = Jinv[..., 0, 0], Jinv[..., 0, 1]
        c, d = Jinv[..., 1, 0], Jinv[..., 1, 1]
        
        n11 = a*a + b*b
        n12 = a*c + b*d
        n22 = c*c + d*d
        
        frobenius_sq = n11 + n22
        discriminant = (frobenius_sq**2) - 4.0 * (det.squeeze()**(-2)) # det(Jinv)^2 = 1/det(J)^2
        # Safe sqrt
        sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
        
        twice_s1s1 = frobenius_sq + sqrt_disc
        s1s1 = 0.5 * twice_s1s1
        s2s2 = 0.5 * (frobenius_sq - sqrt_disc)
        
        # Clamping singular values (s1, s2 are singular values of Jinv)
        # s1 = 1/sigma_min(J). If s1 <= 1, sigma_min(J) >= 1 (upsampling).
        
        # Ellipse parameters
        # We need major/minor axes of the ellipse in INPUT space.
        # If s1s1 <= 1, we clamp to 1.
        major_mag = torch.sqrt(torch.clamp(s1s1, min=1.0))
        minor_mag = torch.sqrt(torch.clamp(s2s2, min=1.0))
        
        # Eigenvector calculation for rotation
        # n - s1^2 I
        t11 = n11 - s1s1
        t22 = n22 - s1s1
        
        # Choose row with largest norm
        use_row1 = (t11*t11 + n12*n12) >= (n12*n12 + t22*t22)
        
        u1 = torch.where(use_row1, -n12, t22) # y component
        u2 = torch.where(use_row1, t11, -n12) # x component ... wait, eigenvector logic is tricky.
        
        # Let's stick to C code logic:
        # temp_u11 = (s1s1-n11)^2 >= (s1s1-n22)^2 ? n12 : s1s1-n22
        # temp_u21 = ... ? s1s1-n11 : n21
        
        diff1 = s1s1 - n11
        diff2 = s1s1 - n22
        cond = (diff1**2) >= (diff2**2)
        
        temp_u11 = torch.where(cond, n12, diff2)
        temp_u21 = torch.where(cond, diff1, n12)
        
        norm = torch.sqrt(temp_u11**2 + temp_u21**2)
        u11 = torch.where(norm > 0, temp_u11/norm, torch.ones_like(norm))
        u21 = torch.where(norm > 0, temp_u21/norm, torch.zeros_like(norm))
        
        # Major/Minor unit vectors
        major_unit_x, major_unit_y = u11, u21
        minor_unit_x, minor_unit_y = -u21, u11
        
        # Ellipse coefficients for distance calc
        # c_major = unit / mag
        c_major_x = major_unit_x / major_mag
        c_major_y = major_unit_y / major_mag
        c_minor_x = minor_unit_x / minor_mag
        c_minor_y = minor_unit_y / minor_mag
        
        # Ellipse area factor for blending
        ellipse_f = major_mag * minor_mag
        theta = 1.0 / ellipse_f
        
        # Decision: Do we need EWA?
        # If twice_s1s1 <= 2.0, it's pure upsampling (or 1:1).
        need_ewa = (twice_s1s1 > 2.0)
        
        # --- DATA GATHERING ---
        # We need a window of pixels around each grid point.
        # For Mitchell (Upsample), we need 4x4.
        # For EWA (Downsample), we need more. 
        # To vectorize, we pick a fixed window size, e.g., 6x6 or 8x8.
        # The C code uses LOHALO_OFFSET_0 = 13 (27x27), which is huge.
        # For a GPU implementation, we'll stick to 6x6 for reasonable performance.
        # If the ellipse is larger than 6x6, it will be clipped (graceful degradation).
        
        win_radius = 3 # 6x6 window
        win_size = 2 * win_radius
        
        # Anchor pixels (top-left of the 4x4 block, or center-ish)
        # GIMP convention: floor(x) is the index of the pixel centered at x+0.5
        # We want the integer part.
        ix = torch.floor(grid[..., 0]).long()
        iy = torch.floor(grid[..., 1]).long()
        
        # Relative coordinates
        fx = grid[..., 0] - (ix.float() + 0.5)
        fy = grid[..., 1] - (iy.float() + 0.5)
        
        # Pad image to handle borders
        img_pad = F.pad(image, (win_radius, win_radius+1, win_radius, win_radius+1), mode='replicate')
        
        # Unfold is hard with arbitrary grids. We use gather.
        # Generate offsets
        oy = torch.arange(-win_radius + 1, win_radius + 1, device=image.device)
        ox = torch.arange(-win_radius + 1, win_radius + 1, device=image.device)
        oy, ox = torch.meshgrid(oy, ox, indexing='ij')
        
        # Shape: (1, 1, win_size, win_size)
        oy = oy.view(1, 1, 1, win_size, win_size)
        ox = ox.view(1, 1, 1, win_size, win_size)
        
        # Neighbor indices: (B, H, W, win_size, win_size)
        # Add padding offset
        n_iy = iy.unsqueeze(-1).unsqueeze(-1) + oy + win_radius
        n_ix = ix.unsqueeze(-1).unsqueeze(-1) + ox + win_radius
        
        # Flatten for gather
        # n_idx = n_iy * W_padded + n_ix
        H_pad, W_pad = img_pad.shape[2], img_pad.shape[3]
        n_iy = torch.clamp(n_iy, 0, H_pad - 1)
        n_ix = torch.clamp(n_ix, 0, W_pad - 1)
        
        # Gather pixels
        # img_pad: (B, C, H_p, W_p)
        # We want (B, C, H_out, W_out, win_size, win_size)
        
        # Flatten spatial dims
        img_flat = img_pad.view(B, C, -1)
        gather_idx = (n_iy * W_pad + n_ix).view(B, 1, H_out * W_out * win_size * win_size)
        gather_idx = gather_idx.expand(-1, C, -1)
        
        pixels = torch.gather(img_flat, 2, gather_idx)
        pixels = pixels.view(B, C, H_out, W_out, win_size, win_size)
        
        # --- MITCHELL (UPSAMPLING) PATH ---
        # Uses 4x4 center of the 6x6 window.
        # Indices in window: 1, 2, 3, 4 (since radius is 3, range is 0..5)
        # Rel coords x_0, y_0 are fx, fy.
        # We need to shift fx, fy to be relative to the pixel centers in the window.
        # Window pixel (j, i) is at location (ix + j - radius + 1 + 0.5, ...)
        # Grid loc is (ix + 0.5 + fx)
        # Delta = Grid - Pixel = fx - (j - radius + 1)
        
        # 1D Weights
        # We need weights for offsets -1, 0, 1, 2 relative to the anchor.
        # In our window (radius 3), the anchor (0,0) is at index (2,2) [0-based 2].
        # Wait, let's align with C code.
        # C code anchor is floor(absolute).
        # Our ix, iy are floor(absolute).
        # C code x_0 = absolute - 0.5 - ix. This matches our fx.
        
        # Mitchell weights for x
        # We calculate weights for the 4 pixels surrounding the sample.
        # If fx > 0, sample is between 0 and 1. Pixels needed: -1, 0, 1, 2.
        # If fx < 0, sample is between -1 and 0. Pixels needed: -2, -1, 0, 1.
        # To simplify, we shift coordinate so we are always in [0, 1].
        # But let's just use the generalized weight function on the grid.
        
        # Coordinates of pixels in window relative to sample point
        # grid_x - pixel_x = fx - (offset)
        # offsets in window: -2, -1, 0, 1, 2, 3
        win_offsets_x = ox.view(1, 1, 1, win_size, win_size).float()
        win_offsets_y = oy.view(1, 1, 1, win_size, win_size).float()
        
        rel_x = fx.unsqueeze(-1).unsqueeze(-1) - win_offsets_x
        rel_y = fy.unsqueeze(-1).unsqueeze(-1) - win_offsets_y
        
        # --- SIGMOIDIZATION (For Mitchell) ---
        # "It appears that it is a bad idea to sigmoidize the transparency channel"
        # We'll apply to all for simplicity, or split if C=4.
        # C code applies inverse_sigmoidal to input, sums, then extended_sigmoidal.
        
        pixels_sigmoid = self.inverse_sigmoid(pixels)
        
        # Mitchell weights are separable.
        # We only need the 4x4 block centered on the target.
        # Find the 4x4 sub-window indices.
        # If fx >= 0, we want indices corresponding to offsets -1, 0, 1, 2.
        # In our window -2, -1, 0, 1, 2, 3. Indices: 1, 2, 3, 4.
        # If fx < 0, we want -2, -1, 0, 1. Indices: 0, 1, 2, 3.
        
        # Actually, let's just compute weights for all 6x6 and zero out the ones not in 4x4.
        # Or better, just compute the 4 weights analytically like C.
        
        # C code logic:
        # ax = abs(x_0) if x_0 >= 0 else -x_0
        # But C code shifts the pointers.
        # Let's stick to the Robidoux 13-flop formula for the 4 weights.
        
        # We calculate the 4 weights for the X dimension and Y dimension.
        # We need to know the shift direction.
        shift_x = (fx >= 0).float() # 1 if positive, 0 if negative
        shift_y = (fy >= 0).float()
        
        # Adjust fx, fy to be in [0, 1] relative to the "base" pixel
        # If fx < 0, we are looking at range [-1, 0]. 
        # The formula assumes ax in [0, 1].
        ax = torch.abs(fx)
        ay = torch.abs(fy)
        
        wx_1, wx_2, wx_3, wx_4 = self.mitchell_weights(ax)
        wy_1, wy_2, wy_3, wy_4 = self.mitchell_weights(ay)
        
        # Map these weights to the window indices
        # If shift_x (fx>=0): weights correspond to offsets -1, 0, 1, 2.
        # In window -2..3, these are indices 1, 2, 3, 4.
        # If not shift_x (fx<0): weights correspond to offsets -2, -1, 0, 1.
        # In window -2..3, these are indices 0, 1, 2, 3.
        
        # Construct 1D weight vectors (win_size)
        w_vec_x = torch.zeros(B, H_out, W_out, win_size, device=image.device)
        w_vec_y = torch.zeros(B, H_out, W_out, win_size, device=image.device)
        
        # Indices
        base_idx = 1 # Center of window is index 2 (offset 0). -1 is index 1.
        # If shift is 0 (negative), we shift index down by 1.
        start_x = base_idx + (shift_x * 0).long() - (1 - shift_x).long() # 1 or 0
        start_y = base_idx + (shift_y * 0).long() - (1 - shift_y).long()
        
        # This is getting messy to vectorize purely.
        # Let's use a mask or scatter.
        idx = torch.arange(win_size, device=image.device).view(1, 1, 1, -1)
        
        # Weights mapping
        # We put [wx_1, wx_2, wx_3, wx_4] into the vector at start_x
        # Since start_x varies per pixel, we use scatter.
        # But scatter requires index to match dims.
        
        # Alternative: Just use the relative coordinate calculation for all 6x6
        # and apply a Mitchell kernel function.
        # Mitchell kernel M(x):
        # if |x| < 1: 7/6|x|^3 - 2|x|^2 + 8/9
        # if 1 <= |x| < 2: -7/18|x|^3 + 2|x|^2 - 10/3|x| + 16/9
        # else 0
        # This is mathematically equivalent to the fast weights.
        
        def mitchell_kernel(x):
            ax = torch.abs(x)
            mask1 = (ax < 1.0).float()
            mask2 = ((ax >= 1.0) & (ax < 2.0)).float()
            
            # Region 1
            v1 = (7.0/6.0)*ax**3 - 2.0*ax**2 + (8.0/9.0)
            # Region 2
            v2 = (-7.0/18.0)*ax**3 + 2.0*ax**2 - (10.0/3.0)*ax + (16.0/9.0)
            
            return v1*mask1 + v2*mask2
            
        w_mitchell_x = mitchell_kernel(rel_x) # (B, H, W, 6, 6)
        w_mitchell_y = mitchell_kernel(rel_y)
        
        # Tensor product weights
        # rel_x varies along last dim (columns), rel_y along second to last (rows)
        # Wait, rel_x is (B, H, W, 6, 6).
        # rel_x depends on ox (columns). rel_y depends on oy (rows).
        w_mitchell = w_mitchell_x * w_mitchell_y
        
        # Apply weights
        mitchell_sum = torch.sum(pixels_sigmoid * w_mitchell.unsqueeze(1), dim=(-1, -2))
        
        # Desigmoidize
        mitchell_val = self.extended_sigmoid(mitchell_sum)
        
        # Handle Alpha (last channel) separately if needed (C code does linear for alpha)
        # For simplicity here, we treat all channels same, or user can split.
        
        # --- EWA (DOWNSAMPLING) PATH ---
        # Uses Robidoux weights based on ellipse distance.
        
        # q1 = s * c_major_x + t * c_major_y
        # q2 = s * c_minor_x + t * c_minor_y
        # s, t are rel_x, rel_y (input space offsets)
        # But wait, rel_x = grid - pixel_center.
        # C code: s = x_0 - j. (x_0 is grid relative to anchor, j is offset).
        # Yes, rel_x is exactly s.
        
        s = rel_x
        t = rel_y
        
        # Expand coefficients to match window
        cmx = c_major_x.unsqueeze(-1).unsqueeze(-1)
        cmy = c_major_y.unsqueeze(-1).unsqueeze(-1)
        cnX = c_minor_x.unsqueeze(-1).unsqueeze(-1)
        cnY = c_minor_y.unsqueeze(-1).unsqueeze(-1)
        
        q1 = s * cmx + t * cmy
        q2 = s * cnX + t * cnY
        
        r2 = q1*q1 + q2*q2
        
        w_ewa = self.robidoux_weight(r2)
        
        # Normalize EWA weights
        total_weight = torch.sum(w_ewa, dim=(-1, -2), keepdim=True) + 1e-8
        
        # EWA is linear light (no sigmoid)
        ewa_sum = torch.sum(pixels * w_ewa.unsqueeze(1), dim=(-1, -2))
        ewa_val = ewa_sum / total_weight.view(B, 1, H_out, W_out)
        
        # --- BLENDING ---
        # beta = (1 - theta) / total_weight if need_ewa else 0
        # newtheta = theta if need_ewa else 1
        # result = newtheta * mitchell + beta * ewa_sum
        # Note: ewa_val is ewa_sum / total_weight.
        # So beta * ewa_sum = (1-theta) * ewa_val.
        # Result = theta * mitchell + (1-theta) * ewa_val.
        
        # Mask for blending
        mask_ewa = need_ewa.float().unsqueeze(1) # (B, 1, H, W)
        
        theta_expanded = theta.unsqueeze(1)
        
        # If not need_ewa, result is mitchell_val.
        # If need_ewa, result is theta*mitchell + (1-theta)*ewa
        
        final_val = torch.where(
            need_ewa.unsqueeze(1),
            theta_expanded * mitchell_val + (1.0 - theta_expanded) * ewa_val,
            mitchell_val
        )
        
        return final_val

class LohaloBasicSampler(nn.Module):
    def __init__(self, contrast=3.38589):
        super(LohaloBasicSampler, self).__init__()
        self.contrast = contrast
        self.sqrt2 = math.sqrt(2.0)
        self.register_buffer('contrast_tensor', torch.tensor(contrast))

    def inverse_sigmoid(self, q):
        sig1 = math.tanh(0.5 * self.contrast * 0.5)
        sig0 = -sig1
        slope = (1.0/sig1 + sig0) * 0.25 * self.contrast
        one_over_slope = 1.0 / slope
        
        mask_low = (q <= 0.0).float()
        mask_high = (q >= 1.0).float()
        mask_mid = 1.0 - mask_low - mask_high
        
        res_low = q * one_over_slope
        res_high = q * one_over_slope + (1.0 - one_over_slope)
        
        ssq = (2.0 * sig1) * q + sig0
        ssq = torch.clamp(ssq, -0.999999, 0.999999)
        res_mid = (2.0 / self.contrast) * torch.atanh(ssq) + 0.5
        
        return res_low * mask_low + res_high * mask_high + res_mid * mask_mid

    def extended_sigmoid(self, q):
        sig1 = math.tanh(0.5 * self.contrast * 0.5)
        slope = (1.0/sig1 - sig1) * 0.25 * self.contrast
        
        mask_low = (q <= 0.0).float()
        mask_high = (q >= 1.0).float()
        mask_mid = 1.0 - mask_low - mask_high
        
        slope_times_q = slope * q
        res_low = slope_times_q
        res_high = slope_times_q + (1.0 - slope)
        
        arg = (0.5 * self.contrast) * q + (-0.25 * self.contrast)
        res_mid = (0.5 / sig1) * torch.tanh(arg) + 0.5
        
        return res_low * mask_low + res_high * mask_high + res_mid * mask_mid

    def robidoux_weight(self, r2):
        a3 = -3.0
        a2 = (45739.0 + 7164.0 * self.sqrt2) / 10319.0
        a0 = (-8926.0 - 14328.0 * self.sqrt2) / 10319.0
        
        minus_inner_root = (-103.0 - 36.0 * self.sqrt2) / (7.0 + 72.0 * self.sqrt2)
        minus_outer_root = -2.0
        
        r = torch.sqrt(r2 + 1e-8)
        
        mask_inner = (r2 < 1.0).float()
        mask_outer = ((r2 >= 1.0) & (r2 < 4.0)).float()
        
        w_inner = r2 * (a3 * r + a2) + a0
        w_outer = (r + minus_inner_root) * (r + minus_outer_root)**2
        
        return (w_inner * mask_inner + w_outer * mask_outer)

    def mitchell_kernel(self, x):
        ax = torch.abs(x)
        mask1 = (ax < 1.0).float()
        mask2 = ((ax >= 1.0) & (ax < 2.0)).float()
        
        v1 = (7.0/6.0)*ax**3 - 2.0*ax**2 + (8.0/9.0)
        v2 = (-7.0/18.0)*ax**3 + 2.0*ax**2 - (10.0/3.0)*ax + (16.0/9.0)
        
        return v1*mask1 + v2*mask2

    def get_jacobian(self, grid):
        grid_pad_x = F.pad(grid, (0, 0, 1, 1, 0, 0), mode='replicate')
        dx = (grid_pad_x[:, :, 2:, :] - grid_pad_x[:, :, :-2, :]) * 0.5
        
        grid_pad_y = F.pad(grid, (0, 0, 0, 0, 1, 1), mode='replicate')
        dy = (grid_pad_y[:, 2:, :, :] - grid_pad_y[:, :-2, :, :]) * 0.5
        
        J = torch.cat([dx.unsqueeze(-1), dy.unsqueeze(-1)], dim=-1)
        return J

    def forward(self, image, grid):
        B, C, H_in, W_in = image.shape
        B, H_out, W_out, _ = grid.shape
        
        J = self.get_jacobian(grid)
        
        det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        det = det.unsqueeze(-1).unsqueeze(-1) + 1e-8
        
        Jinv = torch.zeros_like(J)
        Jinv[..., 0, 0] =  J[..., 1, 1]
        Jinv[..., 0, 1] = -J[..., 0, 1]
        Jinv[..., 1, 0] = -J[..., 1, 0]
        Jinv[..., 1, 1] =  J[..., 0, 0]
        Jinv = Jinv / det
        
        a, b = Jinv[..., 0, 0], Jinv[..., 0, 1]
        c, d = Jinv[..., 1, 0], Jinv[..., 1, 1]
        
        n11 = a*a + b*b
        n12 = a*c + b*d
        n22 = c*c + d*d
        
        frobenius_sq = n11 + n22
        discriminant = (frobenius_sq**2) - 4.0 * (det.squeeze()**(-2))
        sqrt_disc = torch.sqrt(torch.clamp(discriminant, min=0.0))
        
        twice_s1s1 = frobenius_sq + sqrt_disc
        s1s1 = 0.5 * twice_s1s1
        s2s2 = 0.5 * (frobenius_sq - sqrt_disc)
        
        major_mag = torch.sqrt(torch.clamp(s1s1, min=1.0))
        minor_mag = torch.sqrt(torch.clamp(s2s2, min=1.0))
        
        diff1 = s1s1 - n11
        diff2 = s1s1 - n22
        cond = (diff1**2) >= (diff2**2)
        
        temp_u11 = torch.where(cond, n12, diff2)
        temp_u21 = torch.where(cond, diff1, n12)
        
        norm = torch.sqrt(temp_u11**2 + temp_u21**2)
        u11 = torch.where(norm > 0, temp_u11/norm, torch.ones_like(norm))
        u21 = torch.where(norm > 0, temp_u21/norm, torch.zeros_like(norm))
        
        major_unit_x, major_unit_y = u11, u21
        minor_unit_x, minor_unit_y = -u21, u11
        
        c_major_x = major_unit_x / major_mag
        c_major_y = major_unit_y / major_mag
        c_minor_x = minor_unit_x / minor_mag
        c_minor_y = minor_unit_y / minor_mag
        
        ellipse_f = major_mag * minor_mag
        theta = 1.0 / ellipse_f
        
        need_ewa = (twice_s1s1 > 2.0)
        
        win_radius = 3
        win_size = 2 * win_radius
        
        ix = torch.floor(grid[..., 0]).long()
        iy = torch.floor(grid[..., 1]).long()
        
        fx = grid[..., 0] - (ix.float() + 0.5)
        fy = grid[..., 1] - (iy.float() + 0.5)
        
        img_pad = F.pad(image, (win_radius, win_radius+1, win_radius, win_radius+1), mode='replicate')
        
        oy = torch.arange(-win_radius + 1, win_radius + 1, device=image.device)
        ox = torch.arange(-win_radius + 1, win_radius + 1, device=image.device)
        oy, ox = torch.meshgrid(oy, ox, indexing='ij')
        
        oy = oy.view(1, 1, 1, win_size, win_size)
        ox = ox.view(1, 1, 1, win_size, win_size)
        
        n_iy = iy.unsqueeze(-1).unsqueeze(-1) + oy + win_radius
        n_ix = ix.unsqueeze(-1).unsqueeze(-1) + ox + win_radius
        
        H_pad, W_pad = img_pad.shape[2], img_pad.shape[3]
        n_iy = torch.clamp(n_iy, 0, H_pad - 1)
        n_ix = torch.clamp(n_ix, 0, W_pad - 1)
        
        img_flat = img_pad.view(B, C, -1)
        gather_idx = (n_iy * W_pad + n_ix).view(B, 1, H_out * W_out * win_size * win_size)
        gather_idx = gather_idx.expand(-1, C, -1)
        
        pixels = torch.gather(img_flat, 2, gather_idx)
        pixels = pixels.view(B, C, H_out, W_out, win_size, win_size)
        
        win_offsets_x = ox.view(1, 1, 1, win_size, win_size).float()
        win_offsets_y = oy.view(1, 1, 1, win_size, win_size).float()
        
        rel_x = fx.unsqueeze(-1).unsqueeze(-1) - win_offsets_x
        rel_y = fy.unsqueeze(-1).unsqueeze(-1) - win_offsets_y
        
        pixels_sigmoid = self.inverse_sigmoid(pixels)
        
        w_mitchell_x = self.mitchell_kernel(rel_x)
        w_mitchell_y = self.mitchell_kernel(rel_y)
        w_mitchell = w_mitchell_x * w_mitchell_y
        
        mitchell_sum = torch.sum(pixels_sigmoid * w_mitchell.unsqueeze(1), dim=(-1, -2))
        mitchell_val = self.extended_sigmoid(mitchell_sum)
        
        s = rel_x
        t = rel_y
        
        cmx = c_major_x.unsqueeze(-1).unsqueeze(-1)
        cmy = c_major_y.unsqueeze(-1).unsqueeze(-1)
        cnX = c_minor_x.unsqueeze(-1).unsqueeze(-1)
        cnY = c_minor_y.unsqueeze(-1).unsqueeze(-1)
        
        q1 = s * cmx + t * cmy
        q2 = s * cnX + t * cnY
        
        r2 = q1*q1 + q2*q2
        
        w_ewa = self.robidoux_weight(r2)
        
        total_weight = torch.sum(w_ewa, dim=(-1, -2), keepdim=True) + 1e-8
        
        ewa_sum = torch.sum(pixels * w_ewa.unsqueeze(1), dim=(-1, -2))
        ewa_val = ewa_sum / total_weight.view(B, 1, H_out, W_out)
        
        mask_ewa = need_ewa.float().unsqueeze(1)
        theta_expanded = theta.unsqueeze(1)
        
        final_val = torch.where(
            need_ewa.unsqueeze(1),
            theta_expanded * mitchell_val + (1.0 - theta_expanded) * ewa_val,
            mitchell_val
        )
        
        return final_val

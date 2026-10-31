// hydro_adv.cpp — 水文模块进阶内核 (meander / monotonic_repair / carve / rasterize)
// 与 terrain/modules/hydro_module.py 的实现逐位对齐
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
using py::array_t;
using py::array;

namespace {
const double TWO_PI_ = 2.0 * M_PI;

// Python round() 语义 (round-half-to-even), 输入非负
inline int64_t py_round(double x) {
    const double f = std::floor(x);
    const double c = std::ceil(x);
    const double diff = x - f;
    if (diff < 0.5) return (int64_t)f;
    if (diff > 0.5) return (int64_t)c;
    return ((int64_t)f % 2 == 0) ? (int64_t)f : (int64_t)c;
}

inline int64_t py_int_floor(double x) { return (int64_t)std::floor(x); }

// scipy.ndimage.map_coordinates(order=1, mode='nearest') 单点双线性
template <typename TA>
inline double map1(const TA *f, int64_t H, int64_t W,
                   double y, double x) {
    int64_t i0 = (int64_t)std::floor(y);
    int64_t j0 = (int64_t)std::floor(x);
    const double dy = y - (double)i0;
    const double dx = x - (double)j0;
    int64_t i0c = std::min(std::max(i0, (int64_t)0), H - 1);
    int64_t i1c = std::min(std::max(i0 + 1, (int64_t)0), H - 1);
    int64_t j0c = std::min(std::max(j0, (int64_t)0), W - 1);
    int64_t j1c = std::min(std::max(j0 + 1, (int64_t)0), W - 1);
    const double v00 = (double)f[i0c * W + j0c];
    const double v01 = (double)f[i0c * W + j1c];
    const double v10 = (double)f[i1c * W + j0c];
    const double v11 = (double)f[i1c * W + j1c];
    return (1.0 - dy) * ((1.0 - dx) * v00 + dx * v01) +
           dy * ((1.0 - dx) * v10 + dx * v11);
}
}  // namespace

// ---------------------------------------------------------------------------
// monotonic_repair: extract_river_centerlines 内 4 遍单调修复 + 去重
//   acc:(h,w) float32/float64 (模板零拷贝), pp:(n,2) float32, land:(h,w) bool
//   返回修复后 (m,2) float32
// ---------------------------------------------------------------------------
template <typename TA>
array monotonic_repair_impl(const TA *acc_data, int64_t h, int64_t w,
                            array_t<float, py::array::forcecast> pp_py,
                            const bool *land_data,
                            int64_t min_len) {
    if (pp_py.ndim() != 2 || pp_py.shape(1) != 2)
        throw std::runtime_error("monotonic_repair: pp must be (n,2)");

    // pp -> (n,2) double (内部精度; 存 y,x)
    int64_t nn = pp_py.shape(0);
    std::vector<double> yv((size_t)nn), xv((size_t)nn);
    {
        auto pp = pp_py.unchecked<2>();
        for (int64_t i = 0; i < nn; ++i) {
            yv[(size_t)i] = (double)pp(i, 0);
            xv[(size_t)i] = (double)pp(i, 1);
        }
    }

    // NEI 半径表
    int64_t nei_r1[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1},
                            {0, 1},   {1, -1}, {1, 0},  {1, 1}};
    std::vector<std::pair<int64_t, int64_t>> nei_r2, nei_r3;
    for (int64_t dy = -2; dy <= 2; ++dy)
        for (int64_t dx = -2; dx <= 2; ++dx)
            if (!(dy == 0 && dx == 0)) nei_r2.emplace_back(dy, dx);
    for (int64_t dy = -3; dy <= 3; ++dy)
        for (int64_t dx = -3; dx <= 3; ++dx)
            if (!(dy == 0 && dx == 0)) nei_r3.emplace_back(dy, dx);

    const int64_t r1_cnt = 8;
    const int64_t r2_cnt = (int64_t)nei_r2.size();
    const int64_t r3_cnt = (int64_t)nei_r3.size();

    std::vector<double> acc_now((size_t)nn);
    std::vector<uint8_t> keep((size_t)nn);
    std::vector<double> nb(nn);
    const double hypot_ok = 0.12;

    for (int pass = 0; pass < 4; ++pass) {
        // acc_now = map_coordinates(acc, pp.T, order=1, nearest)
        for (int64_t i = 0; i < nn; ++i)
            acc_now[(size_t)i] = map1(acc_data, h, w, yv[(size_t)i], xv[(size_t)i]);
        bool changed = false;
        for (int64_t i = 0; i < nn - 1; ++i) {
            if (acc_now[(size_t)(i + 1)] >= acc_now[(size_t)i] * 0.95) continue;
            const int64_t cy2 = py_round(yv[(size_t)i]);
            const int64_t cx2 = py_round(xv[(size_t)i]);
            double ba = -1.0;
            int64_t by = -1, bx = -1;
            const int64_t cnt_ = (pass <= 1) ? r1_cnt : (pass == 2 ? r2_cnt : r3_cnt);
            const auto &nl = (pass <= 1) ? nullptr : nullptr;  // 占位
            (void)nl;
            for (int64_t k = 0; k < cnt_; ++k) {
                int64_t dy, dx;
                if (pass <= 1) {
                    dy = nei_r1[k][0];
                    dx = nei_r1[k][1];
                } else if (pass == 2) {
                    dy = nei_r2[(size_t)k].first;
                    dx = nei_r2[(size_t)k].second;
                } else {
                    dy = nei_r3[(size_t)k].first;
                    dx = nei_r3[(size_t)k].second;
                }
                const int64_t ny = cy2 + dy;
                const int64_t nx = cx2 + dx;
                if (0 <= ny && ny < h && 0 <= nx && nx < w &&
                        land_data[(size_t)(ny * w + nx)]) {
                    const double a = (double)acc_data[(size_t)(ny * w + nx)];
                    if (a > ba) {
                        ba = a;
                        by = ny;
                        bx = nx;
                    }
                }
            }
            if (ba >= acc_now[(size_t)i] * 0.78 && by >= 0) {
                yv[(size_t)(i + 1)] = (double)by + 0.5;
                xv[(size_t)(i + 1)] = (double)bx + 0.5;
                changed = true;
            }
        }
        // keep 去重
        for (int64_t i = 0; i < nn; ++i) keep[(size_t)i] = 1;
        for (int64_t i = 2; i < nn; ++i) {
            const double dy = yv[(size_t)i] - yv[(size_t)(i - 1)];
            const double dx = xv[(size_t)i] - xv[(size_t)(i - 1)];
            if (std::hypot(dy, dx) < hypot_ok) keep[(size_t)i] = 0;
        }
        int64_t nk = 0;
        for (int64_t i = 0; i < nn; ++i) nk += keep[(size_t)i] ? 1 : 0;
        if (nk < min_len) break;
        // 压缩
        std::vector<double> yv2, xv2;
        yv2.reserve((size_t)nk);
        xv2.reserve((size_t)nk);
        for (int64_t i = 0; i < nn; ++i)
            if (keep[(size_t)i]) {
                yv2.push_back(yv[(size_t)i]);
                xv2.push_back(xv[(size_t)i]);
            }
        yv.swap(yv2);
        xv.swap(xv2);
        const int64_t nn2 = (int64_t)yv.size();
        if (!changed) {
            nn = nn2;  // 已在循环体更新
            break;
        }
        nn = nn2;
    }

    // 最终去重 (len>=4 时)
    if (nn >= 4) {
        for (int64_t i = 0; i < nn; ++i) keep[(size_t)i] = 1;
        for (int64_t i = 2; i < nn; ++i) {
            const double dy = yv[(size_t)i] - yv[(size_t)(i - 1)];
            const double dx = xv[(size_t)i] - xv[(size_t)(i - 1)];
            if (std::hypot(dy, dx) < hypot_ok) keep[(size_t)i] = 0;
        }
        int64_t nk = 0;
        for (int64_t i = 0; i < nn; ++i) nk += keep[(size_t)i] ? 1 : 0;
        if (nk >= min_len) {
            std::vector<double> yv2, xv2;
            yv2.reserve((size_t)nk);
            xv2.reserve((size_t)nk);
            for (int64_t i = 0; i < nn; ++i)
                if (keep[(size_t)i]) {
                    yv2.push_back(yv[(size_t)i]);
                    xv2.push_back(xv[(size_t)i]);
                }
            yv.swap(yv2);
            xv.swap(xv2);
        }
    }

    py::array_t<float> out_py({(py::ssize_t)yv.size(), (py::ssize_t)2});
    auto out = out_py.mutable_unchecked<2>();
    for (size_t i = 0; i < yv.size(); ++i) {
        out((py::ssize_t)i, 0) = (float)yv[i];
        out((py::ssize_t)i, 1) = (float)xv[i];
    }
    return out_py;
}

// dtype 分派包装 (acc 零拷贝, 按 float32/float64 直读)
array monotonic_repair(py::array acc_py,
                       array_t<float, py::array::forcecast> pp_py,
                       array_t<bool, py::array::forcecast> land_py,
                       int64_t min_len, int64_t h, int64_t w) {
    if (acc_py.ndim() != 2 || land_py.ndim() != 2)
        throw std::runtime_error("monotonic_repair: acc/land must be 2D");
    const bool *land_data = land_py.data();
    if (acc_py.dtype().is(py::dtype::of<float>()))
        return monotonic_repair_impl<float>(
            static_cast<const float *>(acc_py.data()), h, w, pp_py,
            land_data, min_len);
    if (acc_py.dtype().is(py::dtype::of<double>()))
        return monotonic_repair_impl<double>(
            static_cast<const double *>(acc_py.data()), h, w, pp_py,
            land_data, min_len);
    throw std::runtime_error("monotonic_repair: acc must be float32 or float64");
}

// ---------------------------------------------------------------------------
// rasterize_polylines: Bresenham 折线栅格化 (返回 (H,W) bool 掩膜)
//   lines: Python list of (n,2) float32 数组 → 用 py::list 接收
// ---------------------------------------------------------------------------
array rasterize_polylines(py::list lines, int64_t H, int64_t W,
                          array_t<bool, py::array::forcecast> land_py) {
    auto land = land_py.unchecked<2>();
    py::array_t<bool> mask_py({(py::ssize_t)H, (py::ssize_t)W});
    auto mask = mask_py.mutable_unchecked<2>();
    for (int64_t i = 0; i < H; ++i)
        for (int64_t j = 0; j < W; ++j) mask(i, j) = false;

    for (py::handle h : lines) {
        auto pp_arr = py::reinterpret_borrow<array_t<float>>(h);
        auto pp = pp_arr.unchecked<2>();
        const int64_t n = pp.shape(0);
        if (n < 2) continue;
        for (int64_t i = 0; i < n - 1; ++i) {
            const int64_t y0 = py_round((double)pp(i, 0));
            const int64_t x0 = py_round((double)pp(i, 1));
            const int64_t y1 = py_round((double)pp(i + 1, 0));
            const int64_t x1 = py_round((double)pp(i + 1, 1));
            int64_t dx = std::abs(x1 - x0);
            int64_t dy = std::abs(y1 - y0);
            const int64_t sx = x0 < x1 ? 1 : -1;
            const int64_t sy = y0 < y1 ? 1 : -1;
            int64_t err = dx - dy;
            int64_t x = x0, y = y0;
            for (;;) {
                if (0 <= y && y < H && 0 <= x && x < W && land(y, x))
                    mask(y, x) = true;
                if (x == x1 && y == y1) break;
                const int64_t e2 = 2 * err;
                if (e2 > -dy) {
                    err -= dy;
                    x += sx;
                }
                if (e2 < dx) {
                    err += dx;
                    y += sy;
                }
            }
        }
    }
    return mask_py;
}

// ---------------------------------------------------------------------------
// carve_path_dig: carve_along_centerlines 单河主体雕刻 (np.maximum.at 写入 dig)
//   dig:(N,) float64 全局缓冲(原地最大写入), pp:(n,2) float64,
//   width_h/depth_h/bend:(n,) float64, norm:(n,2) float64,
//   H,W, cutoff_cross
// ---------------------------------------------------------------------------
void carve_path_dig(array_t<double, py::array::forcecast> dig_py,
                    array_t<double, py::array::forcecast> pp_py,
                    array_t<double, py::array::forcecast> width_h_py,
                    array_t<double, py::array::forcecast> depth_h_py,
                    array_t<double, py::array::forcecast> bend_py,
                    array_t<double, py::array::forcecast> norm_py,
                    int64_t H, int64_t W, double cutoff_cross) {
    auto dig = dig_py.mutable_unchecked<1>();
    auto pp = pp_py.unchecked<2>();
    auto width_h = width_h_py.unchecked<1>();
    auto depth_h = depth_h_py.unchecked<1>();
    auto bend = bend_py.unchecked<1>();
    auto norm = norm_py.unchecked<2>();
    const int64_t n_p = pp.shape(0);
    const double cs_step = 0.3;

    // cs_u = arange(-cutoff_cross, cutoff_cross + cs_step*0.5, cs_step)
    // 与 numpy.arange 一致: count = ceil(len/step), 值 = start + idx*step
    const double stop = cutoff_cross + cs_step * 0.5;
    const int64_t n_cs = (int64_t)std::ceil((stop - (-cutoff_cross)) / cs_step);
    std::vector<double> cs_u((size_t)n_cs);
    for (int64_t k = 0; k < n_cs; ++k) cs_u[(size_t)k] = -cutoff_cross + (double)k * cs_step;

    std::vector<double> rel((size_t)n_cs), prof((size_t)n_cs),
        trough((size_t)n_cs), env((size_t)n_cs), cs_pos((size_t)n_cs),
        py_v((size_t)n_cs), px_v((size_t)n_cs);

    for (int64_t i = 0; i < n_p; ++i) {
        const double cy = pp(i, 0);
        const double cx = pp(i, 1);
        if (!(0.5 <= cy && cy <= (double)H - 1.5 && 0.5 <= cx && cx <= (double)W - 1.5))
            continue;
        const double w_i = width_h(i);
        const double d_i = depth_h(i);
        const double b_i = bend(i);
        const double ny_ = norm(i, 0);
        const double nx_ = norm(i, 1);
        // profile
        double rel_max = 0.0;
        for (int64_t k = 0; k < n_cs; ++k) {
            const double cu = cs_u[(size_t)k];
            const double spos = cu * w_i;
            cs_pos[(size_t)k] = spos;
            const double pyy = cy + spos * ny_;
            const double pxx = cx + spos * nx_;
            py_v[(size_t)k] = pyy;
            px_v[(size_t)k] = pxx;
            prof[(size_t)k] = std::pow(std::abs(cu), 1.35);
            trough[(size_t)k] = std::exp(-((cu - b_i) * (cu - b_i)) / 0.16) * 0.60;
            env[(size_t)k] = std::exp(-(cu * cu) / (2 * 1.05 * 1.05));
            double rv = prof[(size_t)k] - trough[(size_t)k];
            if (rv < 0.0) rv = 0.0;
            rel[(size_t)k] = rv;
            if (rv > rel_max) rel_max = rv;
        }
        const double rel_denom = std::max(rel_max, 1e-8);
        std::vector<int64_t> base_arr((size_t)n_cs);
        std::vector<double> fy_a((size_t)n_cs), fx_a((size_t)n_cs);
        for (int64_t k = 0; k < n_cs; ++k) {
            double rel_k = rel[(size_t)k] / rel_denom;
            const double dig_i = d_i * (1.0 - rel_k) * env[(size_t)k];
            if (dig_i <= 0.0) continue;
            int64_t i0 = py_int_floor(py_v[(size_t)k]);
            int64_t j0 = py_int_floor(px_v[(size_t)k]);
            i0 = std::min(std::max(i0, (int64_t)0), H - 2);
            j0 = std::min(std::max(j0, (int64_t)0), W - 2);
            const double fy = py_v[(size_t)k] - (double)i0;
            const double fx = px_v[(size_t)k] - (double)j0;
            const int64_t barr = i0 * W + j0;
            const double omfy = 1.0 - fy;
            const double omfx = 1.0 - fx;
            double *d = dig.mutable_data(0);
            // np.maximum.at 四邻
            if (dig_i * omfy * omfx > d[barr]) d[barr] = dig_i * omfy * omfx;
            if (dig_i * omfy * fx > d[barr + 1]) d[barr + 1] = dig_i * omfy * fx;
            if (dig_i * fy * omfx > d[barr + W]) d[barr + W] = dig_i * fy * omfx;
            if (dig_i * fy * fx > d[barr + W + 1]) d[barr + W + 1] = dig_i * fy * fx;
        }
    }
}

// ---------------------------------------------------------------------------
// carve_oxbow: 牛轭湖 O(n²) 双循环 (np.maximum.at 写入 dig)
// ---------------------------------------------------------------------------
void carve_oxbow(array_t<double, py::array::forcecast> dig_py,
                 array_t<double, py::array::forcecast> pp_py,
                 int64_t H, int64_t W,
                 array_t<bool, py::array::forcecast> land_py,
                 double max_depth, double oxbow_thresh_px,
                 double oxbow_depth_factor) {
    auto dig = dig_py.mutable_unchecked<1>();
    auto pp = pp_py.unchecked<2>();
    auto land = land_py.unchecked<2>();
    const int64_t n_p = pp.shape(0);
    if (n_p < 40) return;
    std::vector<int64_t> idxs;
    for (int64_t i = 0; i < n_p; i += 3) idxs.push_back(i);
    const int64_t n_i = (int64_t)idxs.size();
    const double d_oxb = max_depth * oxbow_depth_factor;

    for (int64_t a = 0; a < n_i; ++a) {
        const int64_t i = idxs[(size_t)a];
        const int64_t b_lo = a + 6;
        const int64_t b_hi = std::min(a + 30, n_i);
        for (int64_t b = b_lo; b < b_hi; ++b) {
            const int64_t j = idxs[(size_t)b];
            if (j - i < 18) continue;
            const double dy_ = pp(j, 0) - pp(i, 0);
            const double dx_ = pp(j, 1) - pp(i, 1);
            const double d_eu = std::sqrt(dy_ * dy_ + dx_ * dx_);
            if (d_eu > oxbow_thresh_px) continue;
            const double cy = 0.5 * (pp(i, 0) + pp(j, 0));
            const double cx = 0.5 * (pp(i, 1) + pp(j, 1));
            const double ry = std::max(d_eu * 1.1, 3.0);
            const double rx = std::max(d_eu * 0.6, 1.8);
            const double ry3 = std::max(ry, 3.0);
            const double rx3 = std::max(rx, 3.0);
            const int64_t rr0 = py_int_floor(cy - ry3 - 1.0);
            const int64_t rr1 = py_int_floor(cy + ry3 + 2.0 - 1.0) + 1;  // arange stop
            const int64_t cc0 = py_int_floor(cx - rx3 - 1.0);
            const int64_t cc1 = py_int_floor(cx + rx3 + 2.0 - 1.0) + 1;
            for (int64_t yy = rr0; yy < rr1; ++yy) {
                for (int64_t xx = cc0; xx < cc1; ++xx) {
                    if (!(0 <= yy && yy < H && 0 <= xx && xx < W)) continue;
                    if (!land(yy, xx)) continue;
                    const double e = ((double)yy - cy) / std::max(ry, 1e-3);
                    const double f = ((double)xx - cx) / std::max(rx, 1e-3);
                    const double fill = std::exp(-(e * e + f * f) * 1.8) * d_oxb;
                    if (fill <= 1e-5) continue;
                    double *d = dig.mutable_data(0);
                    const int64_t idx_all = yy * W + xx;
                    if (fill > d[idx_all]) d[idx_all] = fill;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// meander_single_river: _meander_single_river + _meander_main_loop 合并
//   path:(n,2) float32, acc_at:(n,) float64, ph:(octaves*5,) float64
//   返回蜿蜒后 (n,2) float32
// ---------------------------------------------------------------------------
namespace {

// 9-tap Hanning 归一化核
void hanning_kern9(double kern[9]) {
    for (int i = 0; i < 9; ++i) kern[i] = 0.5 * (1.0 - std::cos(2.0 * M_PI * (double)i / 8.0));
    double ksum = 0.0;
    for (int i = 0; i < 9; ++i) ksum += kern[i];
    for (int i = 0; i < 9; ++i) kern[i] /= ksum;
}

inline int64_t clip_idx(int64_t i, int64_t n) {
    if (i < 0) return 0;
    if (i >= n) return n - 1;
    return i;
}

}  // namespace

// ---------------------------------------------------------------------------
// meander_single_river: 单条河流完整蜿蜒流程 (Hack 定律 + Hanning 平滑 + 曲率约束)
//   path:(n,2) float32 中心线点列, acc_at:(n,) double 各点流量累积,
//   ph:(h,w) double 相位场; 返回蜿蜒后的 (m,2) float32 中心线
// ---------------------------------------------------------------------------
array meander_single_river(array_t<float, py::array::forcecast> path_py,
                           array_t<double, py::array::forcecast> acc_at_py,
                           double q99,
                           array_t<double, py::array::forcecast> ph_py,
                           double amp, double base_freq,
                           double head_shrink, double tail_shrink,
                           double len_fac,
                           bool scale_adaptive,
                           double min_turn_radius_px,
                           double S_MIN, double S_MAX,
                           double dtheta_max, int64_t clamp_iters,
                           array_t<bool, py::array::forcecast> land_py,
                           bool has_land, int64_t octaves) {
    int64_t n_p = path_py.shape(0);
    if (path_py.ndim() != 2 || path_py.shape(1) != 2)
        throw std::runtime_error("meander_single_river: path must be (n,2)");
    auto path_in = path_py.unchecked<2>();
    if (n_p < 6) {
        py::array_t<float> out({(py::ssize_t)n_p, (py::ssize_t)2});
        auto o = out.mutable_unchecked<2>();
        for (int64_t i = 0; i < n_p; ++i) {
            o(i, 0) = path_in(i, 0);
            o(i, 1) = path_in(i, 1);
        }
        return out;
    }
    auto acc_at = acc_at_py.unchecked<1>();
    auto ph = ph_py.unchecked<1>();

    // path_orig
    std::vector<double> oy0((size_t)n_p), ox0((size_t)n_p);
    for (int64_t i = 0; i < n_p; ++i) {
        oy0[(size_t)i] = (double)path_in(i, 0);
        ox0[(size_t)i] = (double)path_in(i, 1);
    }

    // ---- 前置: dp, seg_len, s, t ----
    std::vector<double> dp_y((size_t)(n_p - 1)), dp_x((size_t)(n_p - 1)),
        seg_len((size_t)(n_p - 1));
    for (int64_t i = 0; i < n_p - 1; ++i) {
        dp_y[(size_t)i] = oy0[(size_t)(i + 1)] - oy0[(size_t)i];
        dp_x[(size_t)i] = ox0[(size_t)(i + 1)] - ox0[(size_t)i];
        seg_len[(size_t)i] = std::sqrt(dp_y[(size_t)i] * dp_y[(size_t)i] +
                                       dp_x[(size_t)i] * dp_x[(size_t)i]);
    }
    std::vector<double> s((size_t)n_p);
    s[0] = 0.0;
    for (int64_t i = 1; i < n_p; ++i) s[(size_t)i] = s[(size_t)(i - 1)] + seg_len[(size_t)(i - 1)];
    double S = s[(size_t)(n_p - 1)];
    if (S < 1e-6) S = 1e-6;
    std::vector<double> t((size_t)n_p);
    for (int64_t i = 0; i < n_p; ++i) t[(size_t)i] = s[(size_t)i] / S;

    // ---- tang, norm_ ----
    std::vector<double> tang_y((size_t)n_p, 0.0), tang_x((size_t)n_p, 0.0);
    if (seg_len[0] > 1e-6) {
        tang_y[0] = dp_y[0] / seg_len[0];
        tang_x[0] = dp_x[0] / seg_len[0];
    }
    if (seg_len[(size_t)(n_p - 2)] > 1e-6) {
        tang_y[(size_t)(n_p - 1)] = dp_y[(size_t)(n_p - 2)] / seg_len[(size_t)(n_p - 2)];
        tang_x[(size_t)(n_p - 1)] = dp_x[(size_t)(n_p - 2)] / seg_len[(size_t)(n_p - 2)];
    }
    for (int64_t i = 1; i < n_p - 1; ++i) {
        const double denom = seg_len[(size_t)(i - 1)] + seg_len[(size_t)i] + 1e-6;
        tang_y[(size_t)i] = (dp_y[(size_t)i] + dp_y[(size_t)(i - 1)]) / denom;
        tang_x[(size_t)i] = (dp_x[(size_t)i] + dp_x[(size_t)(i - 1)]) / denom;
    }
    std::vector<double> norm_y((size_t)n_p), norm_x((size_t)n_p);
    for (int64_t i = 0; i < n_p; ++i) {
        const double tn = std::sqrt(tang_y[(size_t)i] * tang_y[(size_t)i] +
                                    tang_x[(size_t)i] * tang_x[(size_t)i]);
        if (tn > 1e-8) {
            tang_y[(size_t)i] /= tn;
            tang_x[(size_t)i] /= tn;
        }
        norm_y[(size_t)i] = -tang_x[(size_t)i];
        norm_x[(size_t)i] = tang_y[(size_t)i];
    }

    // ---- scale (尺度自适应) ----
    std::vector<double> scale((size_t)n_p, 1.0);
    if (scale_adaptive) {
        double acc_max = 0.0;
        for (int64_t i = 0; i < n_p; ++i)
            if (acc_at(i) > acc_max) acc_max = acc_at(i);
        double a99p = q99;
        const double candidate = acc_max * 0.2;
        if (candidate > a99p) a99p = candidate;
        if (a99p < 10.0) a99p = 10.0;
        double kern[9];
        hanning_kern9(kern);
        std::vector<double> w_ratio((size_t)n_p);
        for (int64_t i = 0; i < n_p; ++i) {
            double a = acc_at(i) / a99p;
            if (a < 1e-4) a = 1e-4;
            if (a > 1.0) a = 1.0;
            w_ratio[(size_t)i] = std::sqrt(a);
        }
        std::vector<double> w_sm((size_t)n_p);
        for (int64_t i = 0; i < n_p; ++i) {
            double ss = 0.0;
            for (int k = 0; k < 9; ++k) {
                const int64_t j = clip_idx(i + k - 4, n_p);
                ss += kern[k] * w_ratio[(size_t)j];
            }
            w_sm[(size_t)i] = ss;
        }
        for (int64_t i = 0; i < n_p; ++i) {
            double sc = 0.10 + 2.55 * w_sm[(size_t)i];
            if (sc < 0.10) sc = 0.10;
            else if (sc > 2.65) sc = 2.65;
            scale[(size_t)i] = sc;
        }
        for (int64_t i = 0; i < n_p; ++i) {
            const double cw = std::pow(std::sin(M_PI * t[(size_t)i]), 0.45);
            scale[(size_t)i] = scale[(size_t)i] * (0.35 + 0.65 * cw);
        }
    }

    // ---- off (多 octave 正弦叠加) ----
    const double freq = base_freq * S;
    std::vector<double> off((size_t)n_p, 0.0);
    for (int64_t o = 0; o < octaves; ++o) {
        for (int64_t k = 0; k < 5; ++k) {
            const double p = ph((size_t)(o * 5 + k));
            const double f = freq * (1.0 + 0.28 * (double)k) *
                             std::pow(2.0, (double)o) * 2.0 * M_PI;
            const double amp_o = 1.0 / std::pow(2.0, (double)o);
            for (int64_t i = 0; i < n_p; ++i)
                off[(size_t)i] += std::sin(t[(size_t)i] * f + p) * amp_o;
        }
    }
    double off_max = 0.0;
    for (int64_t i = 0; i < n_p; ++i)
        if (std::abs(off[(size_t)i]) > off_max) off_max = std::abs(off[(size_t)i]);
    if (off_max < 1e-6) off_max = 1e-6;
    for (int64_t i = 0; i < n_p; ++i) off[(size_t)i] = off[(size_t)i] / off_max * amp;

    // ---- end_w / head_w / tail_w ----
    std::vector<double> end_w((size_t)n_p), head_w((size_t)n_p), tail_w((size_t)n_p);
    for (int64_t i = 0; i < n_p; ++i) {
        const double ti = t[(size_t)i];
        end_w[(size_t)i] = std::pow(std::sin(M_PI * ti), 0.85);
        if (ti < head_shrink) {
            double hw = ti / head_shrink;
            if (hw < 0.0) hw = 0.0;
            else if (hw > 1.0) hw = 1.0;
            head_w[(size_t)i] = std::pow(hw, 1.3);
        } else {
            head_w[(size_t)i] = 1.0;
        }
        if (ti > 1.0 - tail_shrink) {
            double tw = (1.0 - ti) / tail_shrink;
            if (tw < 0.0) tw = 0.0;
            else if (tw > 1.0) tw = 1.0;
            tail_w[(size_t)i] = std::pow(tw, 1.4);
        } else {
            tail_w[(size_t)i] = 1.0;
        }
    }
    double off_abs_max = 0.0;
    for (int64_t i = 0; i < n_p; ++i)
        if (std::abs(off[(size_t)i]) > off_abs_max) off_abs_max = std::abs(off[(size_t)i]);
    std::vector<double> off_base((size_t)n_p);
    if (off_abs_max > 1e-9) {
        for (int64_t i = 0; i < n_p; ++i) off_base[(size_t)i] = off[(size_t)i] / off_abs_max;
    } else {
        for (int64_t i = 0; i < n_p; ++i) off_base[(size_t)i] = off[(size_t)i];
    }

    // ---- path_v1 ----
    std::vector<double> pv_y((size_t)n_p), pv_x((size_t)n_p);
    for (int64_t i = 0; i < n_p; ++i) {
        double amp_at = amp * scale[(size_t)i] * end_w[(size_t)i] *
                        head_w[(size_t)i] * tail_w[(size_t)i] * len_fac;
        if (amp_at < 0.0) amp_at = 0.0;
        const double off_cur = off_base[(size_t)i] * amp_at;
        pv_y[(size_t)i] = oy0[(size_t)i] + off_cur * norm_y[(size_t)i];
        pv_x[(size_t)i] = ox0[(size_t)i] + off_cur * norm_x[(size_t)i];
    }

    // ================= 主循环 =================
    double kern[9];
    hanning_kern9(kern);
    const double dtheta_tight = 35.0 * (M_PI / 180.0);

    // ---- 8 轮 sinuosity 调节, 每轮 clamp_iters 次转弯钳制 ----
    for (int64_t _loop = 0; _loop < 8; ++_loop) {
        for (int64_t _sm = 0; _sm < clamp_iters; ++_sm) {
            std::vector<double> dp1_y((size_t)(n_p - 1)), dp1_x((size_t)(n_p - 1)),
                sl1((size_t)(n_p - 1));
            for (int64_t i = 0; i < n_p - 1; ++i) {
                dp1_y[(size_t)i] = pv_y[(size_t)(i + 1)] - pv_y[(size_t)i];
                dp1_x[(size_t)i] = pv_x[(size_t)(i + 1)] - pv_x[(size_t)i];
                sl1[(size_t)i] = std::sqrt(dp1_y[(size_t)i] * dp1_y[(size_t)i] +
                                           dp1_x[(size_t)i] * dp1_x[(size_t)i]);
            }
            std::vector<double> tg1_y((size_t)n_p, 0.0), tg1_x((size_t)n_p, 0.0);
            if (sl1[0] > 1e-6) {
                tg1_y[0] = dp1_y[0] / sl1[0];
                tg1_x[0] = dp1_x[0] / sl1[0];
            }
            if (sl1[(size_t)(n_p - 2)] > 1e-6) {
                tg1_y[(size_t)(n_p - 1)] = dp1_y[(size_t)(n_p - 2)] / sl1[(size_t)(n_p - 2)];
                tg1_x[(size_t)(n_p - 1)] = dp1_x[(size_t)(n_p - 2)] / sl1[(size_t)(n_p - 2)];
            }
            for (int64_t i = 1; i < n_p - 1; ++i) {
                const double denom = sl1[(size_t)(i - 1)] + sl1[(size_t)i] + 1e-6;
                tg1_y[(size_t)i] = (dp1_y[(size_t)i] + dp1_y[(size_t)(i - 1)]) / denom;
                tg1_x[(size_t)i] = (dp1_x[(size_t)i] + dp1_x[(size_t)(i - 1)]) / denom;
            }
            for (int64_t i = 0; i < n_p; ++i) {
                const double tn = std::sqrt(tg1_y[(size_t)i] * tg1_y[(size_t)i] +
                                            tg1_x[(size_t)i] * tg1_x[(size_t)i]);
                if (tn > 1e-8) {
                    tg1_y[(size_t)i] /= tn;
                    tg1_x[(size_t)i] /= tn;
                }
            }
            // dtheta
            std::vector<double> dtheta((size_t)n_p, 0.0);
            for (int64_t i = 1; i < n_p - 1; ++i) {
                const double a = std::atan2(tg1_y[(size_t)(i - 1)], tg1_x[(size_t)(i - 1)]);
                const double b = std::atan2(tg1_y[(size_t)(i + 1)], tg1_x[(size_t)(i + 1)]);
                double d = b - a;
                if (d > M_PI) d -= TWO_PI_;
                else if (d < -M_PI) d += TWO_PI_;
                dtheta[(size_t)i] = std::abs(d);
            }
            // smooth_mask
            bool any_smooth = false;
            std::vector<uint8_t> need_smooth((size_t)n_p, 0);
            for (int64_t i = 1; i < n_p - 1; ++i) {
                if (dtheta[(size_t)i] > dtheta_max) {
                    any_smooth = true;
                    need_smooth[(size_t)i] = 1;
                } else {
                    const double ds = 0.5 * (sl1[(size_t)(i - 1)] + sl1[(size_t)i]);
                    if (dtheta[(size_t)i] > 0.05) {
                        const double R = ds / std::max(dtheta[(size_t)i], 1e-8);
                        const double w_here = 1.5 * amp * scale[(size_t)i] +
                                              min_turn_radius_px * 0.4;
                        if (R < std::max(min_turn_radius_px, w_here)) {
                            any_smooth = true;
                            need_smooth[(size_t)i] = 1;
                        }
                    }
                }
            }
            if (!any_smooth) break;
            // 平滑 (p_new 基于 path_v1)
            std::vector<double> pn_y(pv_y), pn_x(pv_x);
            for (int64_t i = 1; i < n_p - 1; ++i) {
                if (!need_smooth[(size_t)i]) continue;
                pn_y[(size_t)i] = 0.25 * pv_y[(size_t)(i - 1)] + 0.50 * pv_y[(size_t)i] +
                                  0.25 * pv_y[(size_t)(i + 1)];
                pn_x[(size_t)i] = 0.25 * pv_x[(size_t)(i - 1)] + 0.50 * pv_x[(size_t)i] +
                                  0.25 * pv_x[(size_t)(i + 1)];
            }
            pv_y.swap(pn_y);
            pv_x.swap(pn_x);
        }
        // sinuosity
        double Sarc = 0.0;
        for (int64_t i = 0; i < n_p - 1; ++i) {
            const double dy = pv_y[(size_t)(i + 1)] - pv_y[(size_t)i];
            const double dx = pv_x[(size_t)(i + 1)] - pv_x[(size_t)i];
            Sarc += std::sqrt(dy * dy + dx * dx);
        }
        const double chord_y = pv_y[(size_t)(n_p - 1)] - pv_y[0];
        const double chord_x = pv_x[(size_t)(n_p - 1)] - pv_x[0];
        double chord_len = std::sqrt(chord_y * chord_y + chord_x * chord_x);
        if (chord_len < 1e-6) chord_len = 1e-6;
        const double S_curr = Sarc / chord_len;
        if (S_MIN <= S_curr && S_curr <= S_MAX) break;
        if (Sarc < 1e-6) break;

        // hanning 平滑
        std::vector<double> sm_y((size_t)n_p), sm_x((size_t)n_p);
        for (int64_t i = 0; i < n_p; ++i) {
            double sy = 0.0, sx = 0.0;
            for (int k = 0; k < 9; ++k) {
                const int64_t j = clip_idx(i + k - 4, n_p);
                sy += kern[k] * pv_y[(size_t)j];
                sx += kern[k] * pv_x[(size_t)j];
            }
            sm_y[(size_t)i] = sy;
            sm_x[(size_t)i] = sx;
        }
        // dp2, sl2, tang2, n2
        std::vector<double> dp2_y((size_t)(n_p - 1)), dp2_x((size_t)(n_p - 1)),
            sl2((size_t)(n_p - 1));
        for (int64_t i = 0; i < n_p - 1; ++i) {
            dp2_y[(size_t)i] = pv_y[(size_t)(i + 1)] - pv_y[(size_t)i];
            dp2_x[(size_t)i] = pv_x[(size_t)(i + 1)] - pv_x[(size_t)i];
            sl2[(size_t)i] = std::sqrt(dp2_y[(size_t)i] * dp2_y[(size_t)i] +
                                       dp2_x[(size_t)i] * dp2_x[(size_t)i]);
            if (sl2[(size_t)i] < 1e-6) sl2[(size_t)i] = 1e-6;
        }
        std::vector<double> tang2_y((size_t)n_p, 0.0), tang2_x((size_t)n_p, 0.0);
        tang2_y[0] = dp2_y[0] / sl2[0];
        tang2_x[0] = dp2_x[0] / sl2[0];
        tang2_y[(size_t)(n_p - 1)] = dp2_y[(size_t)(n_p - 2)] / sl2[(size_t)(n_p - 2)];
        tang2_x[(size_t)(n_p - 1)] = dp2_x[(size_t)(n_p - 2)] / sl2[(size_t)(n_p - 2)];
        for (int64_t i = 1; i < n_p - 1; ++i) {
            const double denom = sl2[(size_t)(i - 1)] + sl2[(size_t)i] + 1e-6;
            tang2_y[(size_t)i] = (dp2_y[(size_t)i] + dp2_y[(size_t)(i - 1)]) / denom;
            tang2_x[(size_t)i] = (dp2_x[(size_t)i] + dp2_x[(size_t)(i - 1)]) / denom;
        }
        for (int64_t i = 0; i < n_p; ++i) {
            const double tn = std::sqrt(tang2_y[(size_t)i] * tang2_y[(size_t)i] +
                                        tang2_x[(size_t)i] * tang2_x[(size_t)i]);
            if (tn > 1e-8) {
                tang2_y[(size_t)i] /= tn;
                tang2_x[(size_t)i] /= tn;
            }
        }
        // n2 = [-tang2[:,1], tang2[:,0]]; adjust
        const double factor = (S_curr > S_MAX) ? 0.86 : 1.18;
        for (int64_t i = 0; i < n_p; ++i) {
            const double diff_y = pv_y[(size_t)i] - sm_y[(size_t)i];
            const double diff_x = pv_x[(size_t)i] - sm_x[(size_t)i];
            const double n2_y = -tang2_x[(size_t)i];
            const double n2_x = tang2_y[(size_t)i];
            const double proj = diff_y * n2_y + diff_x * n2_x;
            const double new_off = proj * (factor - 1.0);
            pv_y[(size_t)i] += new_off * n2_y;
            pv_x[(size_t)i] += new_off * n2_x;
        }
    }  // _loop

    // ---- 3 次紧转弯钳制 ----
    for (int64_t _h = 0; _h < 3; ++_h) {
        std::vector<double> dp1_y((size_t)(n_p - 1)), dp1_x((size_t)(n_p - 1)),
            sl1((size_t)(n_p - 1));
        for (int64_t i = 0; i < n_p - 1; ++i) {
            dp1_y[(size_t)i] = pv_y[(size_t)(i + 1)] - pv_y[(size_t)i];
            dp1_x[(size_t)i] = pv_x[(size_t)(i + 1)] - pv_x[(size_t)i];
            sl1[(size_t)i] = std::sqrt(dp1_y[(size_t)i] * dp1_y[(size_t)i] +
                                       dp1_x[(size_t)i] * dp1_x[(size_t)i]);
        }
        std::vector<double> tg1_y((size_t)n_p, 0.0), tg1_x((size_t)n_p, 0.0);
        if (n_p >= 3 && sl1[0] > 1e-8) {
            tg1_y[0] = dp1_y[0] / sl1[0];
            tg1_x[0] = dp1_x[0] / sl1[0];
        }
        if (n_p >= 3 && sl1[(size_t)(n_p - 2)] > 1e-8) {
            tg1_y[(size_t)(n_p - 1)] = dp1_y[(size_t)(n_p - 2)] / sl1[(size_t)(n_p - 2)];
            tg1_x[(size_t)(n_p - 1)] = dp1_x[(size_t)(n_p - 2)] / sl1[(size_t)(n_p - 2)];
        }
        if (n_p >= 4) {
            for (int64_t i = 1; i < n_p - 1; ++i) {
                double denom = sl1[(size_t)(i - 1)] + sl1[(size_t)i];
                if (denom < 1e-6) denom = 1e-6;
                tg1_y[(size_t)i] = (dp1_y[(size_t)i] + dp1_y[(size_t)(i - 1)]) / denom;
                tg1_x[(size_t)i] = (dp1_x[(size_t)i] + dp1_x[(size_t)(i - 1)]) / denom;
            }
        }
        for (int64_t i = 0; i < n_p; ++i) {
            const double tn = std::sqrt(tg1_y[(size_t)i] * tg1_y[(size_t)i] +
                                        tg1_x[(size_t)i] * tg1_x[(size_t)i]);
            if (tn > 1e-8) {
                tg1_y[(size_t)i] /= tn;
                tg1_x[(size_t)i] /= tn;
            }
        }
        bool bad = false;
        for (int64_t i = 1; i < n_p - 1; ++i) {
            const double a = std::atan2(tg1_y[(size_t)(i - 1)], tg1_x[(size_t)(i - 1)]);
            const double b = std::atan2(tg1_y[(size_t)(i + 1)], tg1_x[(size_t)(i + 1)]);
            double d = b - a;
            if (d > M_PI) d -= TWO_PI_;
            else if (d < -M_PI) d += TWO_PI_;
            if (std::abs(d) > dtheta_tight) {
                bad = true;
                break;
            }
        }
        if (!bad) break;
        for (int64_t i = 1; i < n_p - 1; ++i) {
            const double a = std::atan2(tg1_y[(size_t)(i - 1)], tg1_x[(size_t)(i - 1)]);
            const double b = std::atan2(tg1_y[(size_t)(i + 1)], tg1_x[(size_t)(i + 1)]);
            double d = b - a;
            if (d > M_PI) d -= TWO_PI_;
            else if (d < -M_PI) d += TWO_PI_;
            if (std::abs(d) > dtheta_tight) {
                pv_y[(size_t)i] = 0.5 * pv_y[(size_t)(i - 1)] + 0.5 * pv_y[(size_t)(i + 1)];
                pv_x[(size_t)i] = 0.5 * pv_x[(size_t)(i - 1)] + 0.5 * pv_x[(size_t)(i + 1)];
            }
        }
    }

    // ---- 最终 hanning 平滑 ----
    if (n_p >= 9) {
        std::vector<double> tmp_y(pv_y), tmp_x(pv_x);
        for (int64_t i = 0; i < n_p; ++i) {
            double sy = 0.0, sx = 0.0;
            for (int k = 0; k < 9; ++k) {
                const int64_t j = clip_idx(i + k - 4, n_p);
                sy += kern[k] * tmp_y[(size_t)j];
                sx += kern[k] * tmp_x[(size_t)j];
            }
            pv_y[(size_t)i] = sy;
            pv_x[(size_t)i] = sx;
        }
    }

    // ---- 陆地钳制 ----
    if (has_land) {
        auto land = land_py.unchecked<2>();
        const int64_t H_L = land_py.shape(0);
        const int64_t W_L = land_py.shape(1);
        for (int64_t i = 0; i < n_p; ++i) {
            int64_t ir = py_round(pv_y[(size_t)i]);
            int64_t ic = py_round(pv_x[(size_t)i]);
            if (ir < 0) ir = 0;
            else if (ir >= H_L) ir = H_L - 1;
            if (ic < 0) ic = 0;
            else if (ic >= W_L) ic = W_L - 1;
            if (!land(ir, ic)) {
                pv_y[(size_t)i] = pv_y[(size_t)i] * 0.5 + oy0[(size_t)i] * 0.5;
                pv_x[(size_t)i] = pv_x[(size_t)i] * 0.5 + ox0[(size_t)i] * 0.5;
            }
        }
    }
    // clip 负值
    for (int64_t i = 0; i < n_p; ++i) {
        if (pv_y[(size_t)i] < 0.0) pv_y[(size_t)i] = 0.0;
        if (pv_x[(size_t)i] < 0.0) pv_x[(size_t)i] = 0.0;
    }

    py::array_t<float> out({(py::ssize_t)n_p, (py::ssize_t)2});
    auto o = out.mutable_unchecked<2>();
    for (int64_t i = 0; i < n_p; ++i) {
        o(i, 0) = (float)pv_y[(size_t)i];
        o(i, 1) = (float)pv_x[(size_t)i];
    }
    return out;
}

void py_register_hydro_adv(py::module_ &m) {
    m.def("monotonic_repair", &monotonic_repair,
          py::arg("acc"), py::arg("pp"), py::arg("land"),
          py::arg("min_len"), py::arg("h"), py::arg("w"));
    m.def("rasterize_polylines", &rasterize_polylines,
          py::arg("lines"), py::arg("H"), py::arg("W"), py::arg("land"));
    m.def("carve_path_dig", &carve_path_dig,
          py::arg("dig"), py::arg("pp"), py::arg("width_h"), py::arg("depth_h"),
          py::arg("bend"), py::arg("norm"), py::arg("H"), py::arg("W"),
          py::arg("cutoff_cross"));
    m.def("carve_oxbow", &carve_oxbow,
          py::arg("dig"), py::arg("pp"), py::arg("H"), py::arg("W"),
          py::arg("land"), py::arg("max_depth"), py::arg("oxbow_thresh_px"),
          py::arg("oxbow_depth_factor"));
    m.def("meander_single_river", &meander_single_river,
          py::arg("path"), py::arg("acc_at"), py::arg("q99"), py::arg("ph"),
          py::arg("amp"), py::arg("base_freq"),
          py::arg("head_shrink"), py::arg("tail_shrink"), py::arg("len_fac"),
          py::arg("scale_adaptive"),
          py::arg("min_turn_radius_px"), py::arg("S_MIN"), py::arg("S_MAX"),
          py::arg("dtheta_max"), py::arg("clamp_iters"),
          py::arg("land"), py::arg("has_land"), py::arg("octaves"));
}
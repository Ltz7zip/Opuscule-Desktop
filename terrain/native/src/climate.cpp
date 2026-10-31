// climate.cpp — 气候模块内核 (与 terrain/modules/climate_module.py 的 Numba 实现逐位对齐)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <vector>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using py::array_t;
using py::array;

// ---------------------------------------------------------------------------
// _scan_ew_step: 东西向地形雨扫描内核 (2 pass, m 跨 pass 延续)
//   每行 (i) 独立: m[i]/hp 仅依赖本行 -> 转置为 i 外层并行
// ---------------------------------------------------------------------------
array scan_ew_step(array_t<double, py::array::forcecast> hk_py,
                   array_t<double, py::array::forcecast> base_py,
                   array_t<bool, py::array::forcecast> ocean_py,
                   int64_t h, int64_t w, int64_t step) {
    if (hk_py.ndim() != 2 || base_py.ndim() != 2 || ocean_py.ndim() != 2)
        throw std::runtime_error("scan_ew_step: inputs must be 2D");
    auto hk = hk_py.unchecked<2>();
    auto base = base_py.unchecked<2>();
    auto ocean = ocean_py.unchecked<2>();
    py::array_t<float> rain_py({(py::ssize_t)h, (py::ssize_t)w});
    auto rain = rain_py.mutable_unchecked<2>();

    std::vector<float> m((size_t)h, 1.0f);
    for (int pass = 0; pass < 2; ++pass) {
        if (step > 0) {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < h; ++i) {
                const int64_t prev_j = w - 1;
                double hp = hk(i, prev_j);
                float mi = m[(size_t)i];
                for (int64_t j = 0; j < w; ++j) {
                    const double hj_i = hk(i, j);
                    const double hp_i = hp;
                    if (hj_i >= 2.0 && hp_i >= 2.0 && !ocean(i, j)) {
                        if (mi > 0.22f) mi = 0.22f;
                    }
                    double diff = hj_i - hp_i;
                    if (diff < 0.0) diff = 0.0;
                    double oro = 700.0 * diff * (double)mi;
                    if (oro > 2200.0) oro = 2200.0;
                    rain(i, j) = (float)((base(i, j) + oro) *
                                         (0.10 + 0.90 * (double)mi));
                    if (ocean(i, j))
                        mi = 1.0f;
                    else
                        mi = (float)((double)mi * std::exp(-oro / 1500.0) * 0.993);
                    hp = hj_i;
                }
                m[(size_t)i] = mi;
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < h; ++i) {
                const int64_t prev_j = 0;
                double hp = hk(i, prev_j);
                float mi = m[(size_t)i];
                for (int64_t j = w - 1; j >= 0; --j) {
                    const double hj_i = hk(i, j);
                    const double hp_i = hp;
                    if (hj_i >= 2.0 && hp_i >= 2.0 && !ocean(i, j)) {
                        if (mi > 0.22f) mi = 0.22f;
                    }
                    double diff = hj_i - hp_i;
                    if (diff < 0.0) diff = 0.0;
                    double oro = 700.0 * diff * (double)mi;
                    if (oro > 2200.0) oro = 2200.0;
                    rain(i, j) = (float)((base(i, j) + oro) *
                                         (0.10 + 0.90 * (double)mi));
                    if (ocean(i, j))
                        mi = 1.0f;
                    else
                        mi = (float)((double)mi * std::exp(-oro / 1500.0) * 0.993);
                    hp = hj_i;
                }
                m[(size_t)i] = mi;
            }
        }
    }
    return rain_py;
}

// ---------------------------------------------------------------------------
// _scan_ns_step: 南北向地形雨扫描内核 (每列 j 独立 -> 并行 j)
// ---------------------------------------------------------------------------
array scan_ns_step(array_t<double, py::array::forcecast> hk_py,
                   array_t<double, py::array::forcecast> base_py,
                   array_t<bool, py::array::forcecast> ocean_py,
                   int64_t h, int64_t w, int64_t step) {
    if (hk_py.ndim() != 2 || base_py.ndim() != 2 || ocean_py.ndim() != 2)
        throw std::runtime_error("scan_ns_step: inputs must be 2D");
    auto hk = hk_py.unchecked<2>();
    auto base = base_py.unchecked<2>();
    auto ocean = ocean_py.unchecked<2>();
    py::array_t<float> rain_py({(py::ssize_t)h, (py::ssize_t)w});
    auto rain = rain_py.mutable_unchecked<2>();

    std::vector<float> m((size_t)w, 1.0f);
    for (int pass = 0; pass < 2; ++pass) {
        if (step > 0) {
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < w; ++j) {
                const int64_t prev_i = h - 1;
                double hp = hk(prev_i, j);
                float mj = m[(size_t)j];
                for (int64_t i = 0; i < h; ++i) {
                    const double hi_i = hk(i, j);
                    const double hp_i = hp;
                    if (hi_i >= 2.0 && hp_i >= 2.0 && !ocean(i, j)) {
                        if (mj > 0.22f) mj = 0.22f;
                    }
                    double diff = hi_i - hp_i;
                    if (diff < 0.0) diff = 0.0;
                    double oro = 700.0 * diff * (double)mj;
                    if (oro > 2200.0) oro = 2200.0;
                    rain(i, j) = (float)((base(i, j) + oro) *
                                         (0.10 + 0.90 * (double)mj));
                    if (ocean(i, j))
                        mj = 1.0f;
                    else
                        mj = (float)((double)mj * std::exp(-oro / 1500.0) * 0.993);
                    hp = hi_i;
                }
                m[(size_t)j] = mj;
            }
        } else {
#pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < w; ++j) {
                const int64_t prev_i = 0;
                double hp = hk(prev_i, j);
                float mj = m[(size_t)j];
                for (int64_t i = h - 1; i >= 0; --i) {
                    const double hi_i = hk(i, j);
                    const double hp_i = hp;
                    if (hi_i >= 2.0 && hp_i >= 2.0 && !ocean(i, j)) {
                        if (mj > 0.22f) mj = 0.22f;
                    }
                    double diff = hi_i - hp_i;
                    if (diff < 0.0) diff = 0.0;
                    double oro = 700.0 * diff * (double)mj;
                    if (oro > 2200.0) oro = 2200.0;
                    rain(i, j) = (float)((base(i, j) + oro) *
                                         (0.10 + 0.90 * (double)mj));
                    if (ocean(i, j))
                        mj = 1.0f;
                    else
                        mj = (float)((double)mj * std::exp(-oro / 1500.0) * 0.993);
                    hp = hi_i;
                }
                m[(size_t)j] = mj;
            }
        }
    }
    return rain_py;
}

// ---------------------------------------------------------------------------
// _mountain_shield: 背风雨影扫描 (belt_rows 为需扫描的行索引; 原地更新 shield)
//   Numba 版 frac = depth / K_max 为 Python 语义的浮点除法 (int/int -> float64)
// ---------------------------------------------------------------------------
array mountain_shield(array_t<bool, py::array::forcecast> mt_sm_py,
                      array_t<double, py::array::forcecast> hh_py,
                      array_t<double, py::array::forcecast> shield_py,
                      int64_t h, int64_t w,
                      array_t<int64_t, py::array::forcecast> belt_rows_py,
                      int64_t step_sign, int64_t K_max) {
    if (mt_sm_py.ndim() != 2 || hh_py.ndim() != 2 || shield_py.ndim() != 2 ||
        belt_rows_py.ndim() != 1)
        throw std::runtime_error("mountain_shield: bad ndim");
    auto mt_sm = mt_sm_py.unchecked<2>();
    auto hh = hh_py.unchecked<2>();
    auto shield = shield_py.mutable_unchecked<2>();
    auto belt_rows = belt_rows_py.unchecked<1>();

    const int64_t n_rows = belt_rows.shape(0);
#pragma omp parallel for schedule(static)
    for (int64_t ri = 0; ri < n_rows; ++ri) {
        const int64_t i = belt_rows(ri);
        int64_t depth = 0;
        double strength = 0.0;
        if (step_sign > 0) {
            for (int64_t j = 0; j < w; ++j) {
                if (mt_sm(i, j)) {
                    depth = K_max;
                    const double s_here = hh(i, j) * 650.0;
                    if (s_here > strength) strength = s_here;
                } else if (depth > 0) {
                    const double frac = (double)depth / (double)K_max;  // Numba: int/int -> float64
                    const double v = strength * frac;
                    if (v > shield(i, j)) shield(i, j) = v;
                    depth -= 1;
                } else {
                    strength = strength - 25.0;
                    if (strength < 0.0) strength = 0.0;
                }
            }
        } else {
            for (int64_t j = w - 1; j >= 0; --j) {
                if (mt_sm(i, j)) {
                    depth = K_max;
                    const double s_here = hh(i, j) * 650.0;
                    if (s_here > strength) strength = s_here;
                } else if (depth > 0) {
                    const double frac = (double)depth / (double)K_max;
                    const double v = strength * frac;
                    if (v > shield(i, j)) shield(i, j) = v;
                    depth -= 1;
                } else {
                    strength = strength - 25.0;
                    if (strength < 0.0) strength = 0.0;
                }
            }
        }
    }
    return shield_py;
}

// ---------------------------------------------------------------------------
// _median_filter_njit: 单遍中值滤波 (圆盘, 含中心, 越界 nearest; 原地写 out)
// ---------------------------------------------------------------------------
array median_filter_njit(array_t<int8_t, py::array::forcecast> bi_m_py,
                         array_t<bool, py::array::forcecast> land_py,
                         array_t<int8_t, py::array::forcecast> out_py,
                         int64_t r_med, int64_t H, int64_t W) {
    if (bi_m_py.ndim() != 2 || land_py.ndim() != 2 || out_py.ndim() != 2)
        throw std::runtime_error("median_filter_njit: inputs must be 2D");
    auto bi_m = bi_m_py.unchecked<2>();
    auto land = land_py.unchecked<2>();
    auto out = out_py.mutable_unchecked<2>();

    #pragma omp parallel for schedule(static)
    for (int64_t y = 0; y < H; ++y) {
        int8_t vals[64];
        int32_t cnt[256];
        for (int64_t x = 0; x < W; ++x) {
            if (!land(y, x)) {
                out(y, x) = bi_m(y, x);
                continue;
            }
            int n = 0;
            for (int64_t dy = -r_med; dy <= r_med; ++dy) {
                for (int64_t dx = -r_med; dx <= r_med; ++dx) {
                    if (dy * dy + dx * dx > r_med * r_med) continue;
                    const int64_t ny = y + dy;
                    const int64_t nx = x + dx;
                    if (0 <= ny && ny < H && 0 <= nx && nx < W)
                        vals[n] = bi_m(ny, nx);
                    else
                        vals[n] = bi_m(y, x);
                    ++n;
                }
            }
            // 直方图求中值 (int8 值域 256 bins; 替代 O(n^2) 插入排序)
            for (int c = 0; c < 256; ++c) cnt[c] = 0;
            for (int k = 0; k < n; ++k) cnt[(int)vals[k] + 128] += 1;
            const int target = n / 2;  // Numba: vals[n//2] (0-based)
            int acc_cnt = 0;
            int med = -129;
            for (int c = 0; c < 256; ++c) {
                acc_cnt += cnt[c];
                if (acc_cnt > target) {
                    med = c - 128;
                    break;
                }
            }
            out(y, x) = (int8_t)med;
        }
    }
    return out_py;
}

// ---------------------------------------------------------------------------
// _mode_filter_njit: 单遍 mode 滤波 (圆盘, 排除中心, 越界 nearest; 原地写 out)
// ---------------------------------------------------------------------------
array mode_filter_njit(array_t<int8_t, py::array::forcecast> bi_m_py,
                       array_t<bool, py::array::forcecast> land_py,
                       array_t<int8_t, py::array::forcecast> out_py,
                       int64_t r_mode, int64_t H, int64_t W) {
    if (bi_m_py.ndim() != 2 || land_py.ndim() != 2 || out_py.ndim() != 2)
        throw std::runtime_error("mode_filter_njit: inputs must be 2D");
    auto bi_m = bi_m_py.unchecked<2>();
    auto land = land_py.unchecked<2>();
    auto out = out_py.mutable_unchecked<2>();

    #pragma omp parallel for schedule(static)
    for (int64_t y = 0; y < H; ++y) {
        int32_t counts[9];
        for (int64_t x = 0; x < W; ++x) {
            if (!land(y, x)) {
                out(y, x) = bi_m(y, x);
                continue;
            }
            for (int c = 0; c < 9; ++c) counts[c] = 0;
            for (int64_t dy = -r_mode; dy <= r_mode; ++dy) {
                for (int64_t dx = -r_mode; dx <= r_mode; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    if (dy * dy + dx * dx > r_mode * r_mode) continue;
                    const int64_t ny = y + dy;
                    const int64_t nx = x + dx;
                    if (0 <= ny && ny < H && 0 <= nx && nx < W)
                        counts[bi_m(ny, nx)] += 1;
                    else
                        counts[bi_m(y, x)] += 1;
                }
            }
            int best_c = 0;
            int best_cnt = -1;
            for (int c = 0; c < 9; ++c)
                if (counts[c] > best_cnt) {
                    best_cnt = counts[c];
                    best_c = c;
                }
            out(y, x) = (int8_t)best_c;
        }
    }
    return out_py;
}

// ---------------------------------------------------------------------------
// smooth_small_patches: 群系碎片清理 (替代 smooth_biome_labels 内 9 × ndi_label
//   + bincount + isin; 27 次 4096² 连通分量标注 → 单遍 BFS + 面积过滤)
//   对每个陆地像素按 bi_m 值做 4 邻域连通分量, 面积 < min_patch_px 的分量
//   用 mode_bi 对应值覆盖 (语义与 numpy 版完全一致).
//   返回: (H,W) int8 新数组
// ---------------------------------------------------------------------------
array smooth_small_patches(array_t<int8_t, py::array::forcecast> bi_py,
                           array_t<int8_t, py::array::forcecast> mode_py,
                           array_t<bool, py::array::forcecast> land_py,
                           int64_t min_patch_px, int64_t H, int64_t W) {
    const int64_t N = H * W;
    auto bi = bi_py.unchecked<2>();
    auto mode = mode_py.unchecked<2>();
    auto land = land_py.unchecked<2>();

    // comp_id: -1 = 未访问/海洋; >=0 = 分量 id
    std::vector<int32_t> comp_id((size_t)N, -1);
    std::vector<int32_t> comp_size;
    comp_size.reserve(1024);
    std::vector<int64_t> stack;
    stack.reserve(4096);

    // 单遍 BFS: 4 邻域同标签连通
    const int nb[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for (int64_t i = 0; i < N; ++i) {
        if (!land(i / W, i - (i / W) * W) || comp_id[(size_t)i] >= 0) continue;
        const int32_t cid = (int32_t)comp_size.size();
        comp_size.push_back(0);
        // BFS
        stack.clear();
        stack.push_back(i);
        comp_id[(size_t)i] = cid;
        const int8_t tag = bi(i / W, i - (i / W) * W);
        while (!stack.empty()) {
            const int64_t cur = stack.back();
            stack.pop_back();
            comp_size[(size_t)cid] += 1;
            const int64_t cy = cur / W;
            const int64_t cx = cur - cy * W;
            for (int k = 0; k < 4; ++k) {
                const int64_t ny = cy + nb[k][0];
                const int64_t nx = cx + nb[k][1];
                if (0 <= ny && ny < H && 0 <= nx && nx < W &&
                    land(ny, nx)) {
                    const int64_t nb_idx = ny * W + nx;
                    if (comp_id[(size_t)nb_idx] < 0 &&
                        bi(ny, nx) == tag) {
                        comp_id[(size_t)nb_idx] = cid;
                        stack.push_back(nb_idx);
                    }
                }
            }
        }
    }

    py::array_t<int8_t> out_py({(py::ssize_t)H, (py::ssize_t)W});
    auto out = out_py.mutable_unchecked<2>();
#pragma omp parallel for schedule(static)
    for (int64_t y = 0; y < H; ++y) {
        for (int64_t x = 0; x < W; ++x) {
            const int32_t cid = comp_id[(size_t)(y * W + x)];
            if (cid >= 0 && comp_size[(size_t)cid] < min_patch_px)
                out(y, x) = mode(y, x);
            else
                out(y, x) = bi(y, x);
        }
    }
    return out_py;
}

void py_register_climate(py::module_ &m) {
    m.def("scan_ew_step", &scan_ew_step,
          py::arg("hk"), py::arg("base"), py::arg("ocean"),
          py::arg("h"), py::arg("w"), py::arg("step"));
    m.def("scan_ns_step", &scan_ns_step,
          py::arg("hk"), py::arg("base"), py::arg("ocean"),
          py::arg("h"), py::arg("w"), py::arg("step"));
    m.def("mountain_shield", &mountain_shield,
          py::arg("mt_sm"), py::arg("hh"), py::arg("shield"),
          py::arg("h"), py::arg("w"),
          py::arg("belt_rows"), py::arg("step_sign"), py::arg("K_max"));
    m.def("median_filter_njit", &median_filter_njit,
          py::arg("bi_m"), py::arg("land"), py::arg("out"),
          py::arg("r_med"), py::arg("H"), py::arg("W"));
    m.def("mode_filter_njit", &mode_filter_njit,
          py::arg("bi_m"), py::arg("land"), py::arg("out"),
          py::arg("r_mode"), py::arg("H"), py::arg("W"));
    m.def("smooth_small_patches", &smooth_small_patches,
          py::arg("bi"), py::arg("mode"), py::arg("land"),
          py::arg("min_patch_px"), py::arg("H"), py::arg("W"));
}
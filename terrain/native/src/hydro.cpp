// hydro.cpp — 水文模块内核 (与 terrain/modules/hydro_module.py 的 Numba 实现逐位对齐)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <cmath>
#include <cstdint>
#include <deque>
#include <queue>
#include <vector>
#include <stdexcept>

namespace py = pybind11;
using py::array_t;
using py::array;

// ---------------------------------------------------------------------------
// _flow_acc_kahn_njit: D8 流量累积 (Kahn 拓扑排序, 单遍 O(N))
//   e:(N,) float32    已加噪声扰动的高程 (1D)
//   source:(N,) float64 每个格点水源值 (被原地修改为累积值)
// ---------------------------------------------------------------------------
array flow_acc_kahn(array_t<float, py::array::forcecast> e_py,
                    array_t<double, py::array::forcecast> acc_py,
                    int64_t h, int64_t w) {
    const int64_t n = h * w;
    if (e_py.ndim() != 1 || acc_py.ndim() != 1)
        throw std::runtime_error("flow_acc_kahn: inputs must be 1D");
    if (e_py.shape(0) != n)
        throw std::runtime_error("flow_acc_kahn: e length mismatch");
    auto e = e_py.unchecked<1>();
    auto acc = acc_py.mutable_unchecked<1>();

    // ---- 1. D8 接收者 ----
    std::vector<int32_t> recv((size_t)n, -1);
    std::vector<float> best_s((size_t)n, 1e-9f);

    for (int64_t i = 0; i < n; ++i) {
        const int64_t y = i / w;
        const int64_t x = i - y * w;
        const float ei = e(i);
        // 对角 (distance 1/sqrt(2)*... 在 Numba 中用 0.70710678 双精度因子)
        for (int64_t dy : {-1, 1}) {
            for (int64_t dx : {-1, 1}) {
                const int64_t ny = y + dy;
                const int64_t nx = x + dx;
                if (0 <= ny && ny < h && 0 <= nx && nx < w) {
                    const int64_t j = ny * w + nx;
                    // Numba: (ei - e[j]) 在 float32 域相减, 再乘 double 0.70710678
                    const double s = (double)(ei - e(j)) * 0.70710678;
                    if (s > (double)best_s[i]) {
                        best_s[i] = (float)s;
                        recv[i] = (int32_t)j;
                    }
                }
            }
        }
        // 直邻
        for (int64_t dy : {-1, 1}) {
            for (int64_t dx : {-1, 1}) {
                const int64_t ny1 = y + dy;
                if (0 <= ny1 && ny1 < h) {
                    const int64_t j1 = ny1 * w + x;
                    const float s1 = ei - e(j1);
                    if (s1 > best_s[i]) {
                        best_s[i] = s1;
                        recv[i] = (int32_t)j1;
                    }
                }
                const int64_t nx2 = x + dx;
                if (0 <= nx2 && nx2 < w) {
                    const int64_t j2 = y * w + nx2;
                    const float s2 = ei - e(j2);
                    if (s2 > best_s[i]) {
                        best_s[i] = s2;
                        recv[i] = (int32_t)j2;
                    }
                }
            }
        }
    }

    // ---- 2. 入度 ----
    std::vector<int32_t> in_degree((size_t)n, 0);
    for (int64_t i = 0; i < n; ++i) {
        const int32_t r = recv[i];
        if (r >= 0) in_degree[(size_t)r] += 1;
    }

    // ---- 3. Kahn 拓扑排序累积 ----
    std::vector<int64_t> queue((size_t)n);
    int64_t head = 0, tail = 0;
    for (int64_t i = 0; i < n; ++i)
        if (in_degree[(size_t)i] == 0) queue[(size_t)tail++] = i;

    while (head < tail) {
        const int64_t i = queue[(size_t)head++];
        const int32_t r = recv[(size_t)i];
        if (r >= 0) {
            acc((size_t)r) += acc((size_t)i);
            in_degree[(size_t)r] -= 1;
            if (in_degree[(size_t)r] == 0) queue[(size_t)tail++] = (int64_t)r;
        }
    }

    return acc_py;
}

// ---------------------------------------------------------------------------
// _fill_depressions_pf: Priority-Flood+eps (Barnes 2014) — 自实现二叉堆
//   elev:(H,W) float32, eps:double, 返回 filled:(H,W) float32
// ---------------------------------------------------------------------------
array fill_depressions_pf(array_t<float, py::array::forcecast> elev_py,
                          double eps, int64_t H, int64_t W) {
    const int64_t N = H * W;
    if (elev_py.ndim() != 2)
        throw std::runtime_error("fill_depressions_pf: elev must be 2D");
    auto elev_flat = elev_py.unchecked<2>();

    std::vector<double> filled((size_t)N);
    for (int64_t i = 0; i < N; ++i) {
        const int64_t y = i / W;
        const int64_t x = i - y * W;
        filled[(size_t)i] = (double)elev_flat(y, x);
    }
    std::vector<uint8_t> visited((size_t)N, 0);

    // 最小堆。注意: 比较器只按高度 (double) 比较 —— 与 Numba 语义一致
    // (高度相等不交换)。std::pair 默认比较会在高度相等时比较 int64 index,
    // 而 fill 过程产生海量相同高度 (h0+eps), 额外 index 比较是主要开销
    // (采样显示 83% 时间在 pop_heap 下沉)。
    using Item = std::pair<double, int64_t>;
    struct HeightOnly {
        bool operator()(const Item &a, const Item &b) const {
            return a.first > b.first;  // min-heap, 仅按高度
        }
    };
    std::priority_queue<Item, std::vector<Item>, HeightOnly> pq;

    // ---- 1. 推入所有边界格点 ----
    for (int64_t x = 0; x < W; ++x) {
        const int64_t idx = x;
        pq.emplace(filled[(size_t)idx], idx);
        visited[(size_t)idx] = 1;
    }
    for (int64_t x = 0; x < W; ++x) {
        const int64_t idx = (H - 1) * W + x;
        pq.emplace(filled[(size_t)idx], idx);
        visited[(size_t)idx] = 1;
    }
    for (int64_t y = 1; y < H - 1; ++y) {
        int64_t idx = y * W;
        pq.emplace(filled[(size_t)idx], idx);
        visited[(size_t)idx] = 1;
        idx = y * W + (W - 1);
        pq.emplace(filled[(size_t)idx], idx);
        visited[(size_t)idx] = 1;
    }

    // ---- 2. 主循环: 弹堆 -> 处理 8 邻域 ----
    while (!pq.empty()) {
        const double h0 = pq.top().first;
        const int64_t i0 = pq.top().second;
        pq.pop();

        const int64_t y0 = i0 / W;
        const int64_t x0 = i0 - y0 * W;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dy == 0 && dx == 0) continue;
                const int64_t ny = y0 + dy;
                const int64_t nx = x0 + dx;
                if (0 <= ny && ny < H && 0 <= nx && nx < W) {
                    const int64_t nidx = ny * W + nx;
                    if (!visited[(size_t)nidx]) {
                        visited[(size_t)nidx] = 1;
                        if (filled[(size_t)nidx] < h0 + eps)
                            filled[(size_t)nidx] = h0 + eps;
                        pq.emplace(filled[(size_t)nidx], nidx);
                    }
                }
            }
        }
    }

    py::array_t<float> out_py({(py::ssize_t)H, (py::ssize_t)W});
    auto out = out_py.mutable_unchecked<2>();
    for (int64_t i = 0; i < N; ++i) {
        const int64_t y = i / W;
        const int64_t x = i - y * W;
        out(y, x) = (float)filled[(size_t)i];
    }
    return out_py;
}

// ---------------------------------------------------------------------------
// _trace_centerline: 河流中心线 D8 贪心追踪 (单连通分量)
//   acc:(h,w) float32/float64 (按 dtype 模板, 零拷贝直读),
//   labels:(h,w) int32, visited:(h,w) int32 原地标记
//   返回 path:(n,2) float32 (源在上游, 汇于下游)
// ---------------------------------------------------------------------------
template <typename TA>
array trace_centerline_impl(const TA *acc_data, int64_t h, int64_t w,
                            const int32_t *labels, int64_t comp_i,
                            int64_t y0, int64_t x0, int64_t max_steps,
                            int32_t *visited) {
    std::vector<float> path_y((size_t)max_steps), path_x((size_t)max_steps);
    int64_t cy = y0, cx = x0;
    int64_t n = 0;
    path_y[(size_t)n] = (float)cy;
    path_x[(size_t)n] = (float)cx;
    ++n;
    visited[(size_t)(cy * w + cx)] = (int32_t)comp_i;

    for (int64_t step = 0; step < max_steps - 1; ++step) {
        double best_a = -1.0;
        int64_t best_y = -1, best_x = -1;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dy == 0 && dx == 0) continue;
                const int64_t ny = cy + dy;
                const int64_t nx = cx + dx;
                if (0 <= ny && ny < h && 0 <= nx && nx < w &&
                    visited[(size_t)(ny * w + nx)] != (int32_t)comp_i &&
                    labels[(size_t)(ny * w + nx)] == (int32_t)comp_i) {
                    const double a = (double)acc_data[(size_t)(ny * w + nx)];
                    if (a > best_a) {
                        best_a = a;
                        best_y = ny;
                        best_x = nx;
                    }
                }
            }
        }
        if (best_y < 0) break;
        cy = best_y;
        cx = best_x;
        path_y[(size_t)n] = (float)cy;
        path_x[(size_t)n] = (float)cx;
        ++n;
        visited[(size_t)(cy * w + cx)] = (int32_t)comp_i;
    }

    // 反转 path (源在上游, 汇于下游)
    py::array_t<float> result_py({(py::ssize_t)n, (py::ssize_t)2});
    auto result = result_py.mutable_unchecked<2>();
    for (int64_t i = 0; i < n; ++i) {
        result(i, 0) = path_y[(size_t)(n - 1 - i)];
        result(i, 1) = path_x[(size_t)(n - 1 - i)];
    }
    return result_py;
}

// ---------------------------------------------------------------------------
// trace_centerline: pybind 入口 —— 按 acc 的 dtype (float32/float64) 分派模板
//   零拷贝直读 (py::array, 避免 forcecast 无条件拷贝导致全流程 +34s)
// ---------------------------------------------------------------------------
array trace_centerline(py::array acc_py, py::array labels_py,
                       int64_t comp_i, int64_t h, int64_t w,
                       int64_t y0, int64_t x0, int64_t max_steps,
                       py::array visited_py) {
    if (acc_py.ndim() != 2 || labels_py.ndim() != 2 || visited_py.ndim() != 2)
        throw std::runtime_error("trace_centerline: inputs must be 2D");
    if (!labels_py.dtype().is(py::dtype::of<int32_t>()) ||
        !visited_py.dtype().is(py::dtype::of<int32_t>()))
        throw std::runtime_error("trace_centerline: labels/visited must be int32");
    const int32_t *labels = static_cast<const int32_t *>(labels_py.data());
    int32_t *visited = static_cast<int32_t *>(visited_py.mutable_data());
    if (acc_py.dtype().is(py::dtype::of<float>()))
        return trace_centerline_impl<float>(static_cast<const float *>(acc_py.data()),
                                            h, w, labels, comp_i, y0, x0, max_steps, visited);
    if (acc_py.dtype().is(py::dtype::of<double>()))
        return trace_centerline_impl<double>(static_cast<const double *>(acc_py.data()),
                                             h, w, labels, comp_i, y0, x0, max_steps, visited);
    throw std::runtime_error("trace_centerline: acc must be float32 or float64");
}

// ---------------------------------------------------------------------------
// fill_depressions_bf: Barnes et al. 2014 "Priority-Flood" improved variation —
//   用普通 FIFO 队列填充洼地 (O(m log m), m = 非洼地格点数)。
//   堆弹出格点展开邻居时: 高度低于水位+eps 的格点(洼地)直接入 FIFO 并
//   filled = 父值 + eps (与标准 Priority-Flood+eps 的 ε 链语义一致);
//   堆与 FIFO 按水位交错推进保证全局处理顺序与标准版一致 -> 输出逐位一致,
//   洼地格点免堆。论文实测比标准版快至多 37%。
// ---------------------------------------------------------------------------
array fill_depressions_bf(array_t<float, py::array::forcecast> elev_py,
                          double eps, int64_t H, int64_t W) {
    const int64_t N = H * W;
    auto elev = elev_py.unchecked<2>();

    std::vector<double> filled((size_t)N);
    for (int64_t i = 0; i < N; ++i) {
        const int64_t y = i / W;
        const int64_t x = i - y * W;
        filled[(size_t)i] = (double)elev(y, x);
    }
    std::vector<uint8_t> visited((size_t)N, 0);

    using Item = std::pair<double, int64_t>;
    struct HeightOnly {
        bool operator()(const Item &a, const Item &b) const {
            return a.first > b.first;
        }
    };
    std::priority_queue<Item, std::vector<Item>, HeightOnly> heap;
    std::deque<int64_t> fifo;

    auto expand = [&](int64_t cur, double hcur) {
        const int64_t cy = cur / W;
        const int64_t cx = cur - cy * W;
        for (int64_t dy = -1; dy <= 1; ++dy) {
            for (int64_t dx = -1; dx <= 1; ++dx) {
                if (dy == 0 && dx == 0) continue;
                const int64_t ny = cy + dy;
                const int64_t nx = cx + dx;
                if (0 <= ny && ny < H && 0 <= nx && nx < W) {
                    const int64_t nidx = ny * W + nx;
                    if (visited[(size_t)nidx]) continue;
                    visited[(size_t)nidx] = 1;
                    const double eb = (double)elev(ny, nx);
                    if (eb < hcur + eps) {   // 洼地: 走 FIFO 免堆
                        filled[(size_t)nidx] = hcur + eps;
                        fifo.push_back(nidx);
                    } else {                  // 高于水位: 入堆
                        filled[(size_t)nidx] = eb;
                        heap.emplace(eb, nidx);
                    }
                }
            }
        }
    };

    // ---- 1. 推入所有边界格点 ----
    for (int64_t x = 0; x < W; ++x) {
        const int64_t idx = x;
        heap.emplace(filled[(size_t)idx], idx);
        visited[(size_t)idx] = 1;
    }
    for (int64_t x = 0; x < W; ++x) {
        const int64_t idx = (H - 1) * W + x;
        heap.emplace(filled[(size_t)idx], idx);
        visited[(size_t)idx] = 1;
    }
    for (int64_t y = 1; y < H - 1; ++y) {
        int64_t idx = y * W;
        heap.emplace(filled[(size_t)idx], idx);
        visited[(size_t)idx] = 1;
        idx = y * W + (W - 1);
        heap.emplace(filled[(size_t)idx], idx);
        visited[(size_t)idx] = 1;
    }

    // ---- 2. 主循环: 堆与 FIFO 按水位(值)交错推进 ----
    //     FIFO 头的水位 == 标准版中该格点入堆后的堆内位置, 故全局处理顺序
    //     与标准 Priority-Flood 完全一致 -> filled 逐位一致, 而洼地格点免堆。
    while (true) {
        const bool have_h = !heap.empty();
        const bool have_f = !fifo.empty();
        if (!have_h && !have_f) break;
        if (have_f && (!have_h ||
                       filled[(size_t)fifo.front()] <= heap.top().first)) {
            const int64_t f = fifo.front();
            fifo.pop_front();
            expand(f, filled[(size_t)f]);
        } else {
            const double h0 = heap.top().first;
            const int64_t c = heap.top().second;
            heap.pop();
            expand(c, h0);
        }
    }

    py::array_t<float> out_py({(py::ssize_t)H, (py::ssize_t)W});
    auto out = out_py.mutable_unchecked<2>();
    for (int64_t i = 0; i < N; ++i) {
        const int64_t y = i / W;
        const int64_t x = i - y * W;
        out(y, x) = (float)filled[(size_t)i];
    }
    return out_py;
}

void py_register_hydro(py::module_ &m) {
    m.def("flow_acc_kahn", &flow_acc_kahn,
          py::arg("e"), py::arg("acc"), py::arg("h"), py::arg("w"));
    m.def("fill_depressions_pf", &fill_depressions_pf,
          py::arg("elev"), py::arg("eps"), py::arg("H"), py::arg("W"));
    m.def("fill_depressions_bf", &fill_depressions_bf,
          py::arg("elev"), py::arg("eps"), py::arg("H"), py::arg("W"));
    m.def("trace_centerline", &trace_centerline,
          py::arg("acc"), py::arg("labels"), py::arg("comp_i"),
          py::arg("h"), py::arg("w"), py::arg("y0"), py::arg("x0"),
          py::arg("max_steps"), py::arg("visited"));
}
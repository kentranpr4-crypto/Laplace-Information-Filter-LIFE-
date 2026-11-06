// autodiff_smoke.cpp - forward-mode derivative check with autodiff (built only when
// A2_PROBE_AUTODIFF_VERSION is set). f(x, y) = x^2 * y + sin(y): df/dx = 2xy, df/dy = x^2 + cos(y).
//
// Note: the autodiff Eigen integration header (autodiff/forward/dual/eigen.hpp) does NOT compile
// against Eigen 5.0.1 with autodiff 1.1.2 (Eigen::IndexedView template mismatch). It is therefore
// deliberately not included here; verify-env.sh reports that separately as its own check.
#include <autodiff/forward/dual.hpp>

#include <cmath>
#include <print>

int main() {
    using autodiff::dual;
    using autodiff::derivative;

    const double xv = 1.5;
    const double yv = 0.7;

    dual x = xv;
    x.grad = 1.0;  // seed d/dx
    dual y = yv;
    dual f = x * x * y + sin(y);
    const double dfdx = derivative(f);

    dual x2 = xv;
    dual y2 = yv;
    y2.grad = 1.0;  // seed d/dy
    dual f2 = x2 * x2 * y2 + sin(y2);
    const double dfdy = derivative(f2);

    const double expected_dfdx = 2.0 * xv * yv;
    const double expected_dfdy = xv * xv + std::cos(yv);
    std::println("autodiff dfdx={:.12f} (expected {:.12f})", dfdx, expected_dfdx);
    std::println("autodiff dfdy={:.12f} (expected {:.12f})", dfdy, expected_dfdy);
    const bool ok = std::abs(dfdx - expected_dfdx) < 1e-12 && std::abs(dfdy - expected_dfdy) < 1e-12;
    std::println("A2SMOKE check {} autodiff forward-mode derivatives match closed form", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// doctest_probe.cpp - proves the doctest package from the toolchain prefix is usable with C++23.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <print>

TEST_CASE("doctest is linked and C++23 std::print/println are available") {
    INFO("doctest version macro: " DOCTEST_VERSION_STR);
    std::println("doctest {} (C++23 std::println ok)", DOCTEST_VERSION_STR);
    CHECK(__cplusplus >= 202302L);
    REQUIRE(1 + 1 == 2);
}

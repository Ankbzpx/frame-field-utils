if(TARGET igl::core)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG v2.4.0
    # FIXME: use UPDATE_DISCONNECTED
    # https://gitlab.kitware.com/cmake/cmake/-/issues/21146
    PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/libigl_dc_disable_parallel_for.patch || true
)
FetchContent_MakeAvailable(libigl)

if(TARGET scsdir)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    scs
    GIT_REPOSITORY https://github.com/cvxgrp/scs.git
    GIT_TAG 3.2.3
    # https://gitlab.kitware.com/cmake/cmake/-/issues/21146
    PATCH_COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/cmake/scs_uninstall_conflict_fix.patch || true
)
FetchContent_MakeAvailable(scs)

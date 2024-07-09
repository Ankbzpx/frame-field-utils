if(TARGET TBB::tbb)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    tbb
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG v2021.13.0
)
set(TBB_TEST OFF)
FetchContent_MakeAvailable(tbb)

cmake_minimum_required(VERSION 3.22.1)

project("smartturn-native")

add_library(
    smartturn-native
    SHARED
    native-lib.cpp
)

find_library(
    log-lib
    log
)

target_link_libraries(
    smartturn-native
    ${log-lib}
    m
)

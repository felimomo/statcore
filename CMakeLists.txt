cmake_minimum_required(VERSION 3.15)

project(statcore VERSION 0.1
                DESCRIPTION "Some statistical tooling."
                LANGUAGES CXX)

add_library(statcore STATIC src/main.cpp)
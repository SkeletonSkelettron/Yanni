# It's important to specify the minimum CMake version upfront required by
# CMakeLists.txt. This is so that a user can clearly understand the reason the 
# build will fail before the build actually occurs, instead of searching for the
# CMake function that was used that is causing the failure.
cmake_minimum_required(VERSION 3.1)

# Set the global package-wide C++ standard. This will be inherited by all
# targets specified in the project. One can also specify the C++ standard in a
# target-specific manner, using:
#   set_target_properties(foo PROPERTIES CXX_STANDARD 11)
#   target_compile_features(foo PUBLIC cxx_std_14)
# together.
set(CMAKE_CXX_STANDARD 11)

# Set the project name and version number. This allows for a user of your
# library or tool to specify a particular version when they include it, as in 
#   find_package(Yanni 1.0 REQUIRED)
project(Yanni VERSION 1.0 LANGUAGES CXX CUDA)
set(Yanni_VERSION 1.0)
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CUDA_SEPARABLE_COMPILATION ON)
 
# Function to prepend the subdirectory to source files in subdirectories
FUNCTION(PREPEND var )
   SET(listVar "")
   FOREACH(f ${${var}})
       LIST(APPEND listVar "${CMAKE_CURRENT_SOURCE_DIR}/${f}")
   ENDFOREACH(f)
   SET(${var} "${listVar}" PARENT_SCOPE)
ENDFUNCTION(PREPEND)

# After a normal build, we can specify the location of various outputs of the
# build. We put executables and static libraries outside the build directory in
# bin/ and lib/, respectively.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib")
 
# Include source code and headers. This calls the CMakeLists.txt in each
# subdirectory. These can define their own libraries, executables, etc. as targets, 
# but here we define all exportable targets in the root CMakeLists.txt.
add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(cuda)
# Add the library Yanni as a target, with the contents of src/ and include/
# as dependencies.
add_library(Yanni STATIC ${Yanni_SRC} ${Yanni_INC})
add_executable(YanniExec Yanni.cpp)
# These variables slightly modify the install location to allow for version
# specific installations.
set(Yanni_INCLUDE_DEST "include/Yanni-${Yanni_VERSION}")
set(Yanni_LIB_DEST "lib/Yanni-${Yanni_VERSION}")

CUDA_ADD_LIBRARY(yanniCudaLib SHARED ${Yanni_CUDA_INC})
set_target_properties(yanniCudaLib PROPERTIES CUDA_ARCHITECTURES "75")
target_link_libraries(yanniCudaLib ${CUDA_LIBRARIES})

# generator expressions are needed for the include directories, since installing headers changes the include path
# Specify that Yanni requires the files located in the include/ directory at
# compile time. This would normally look like 
#   target_include_directories(Yanni PUBLIC include/)
# PUBLIC means that other libraries including Yanni should also include the
# directory include/.
# However, there is a catch. If we are installing the project in
# CMAKE_INSTALL_PREFIX, we can't specify include/ in the build directory: we have 
# copied the contents of include to CMAKE_INSTALL_PREFIX/include and we would
# like  other projects to include this directory instead of include/. The follow
# CMake command handles this. $<BUILD_INTERFACE:...> and
# $<INSTALL_INTERFACE:...> are macros whose values change depending on if we are
# simply building the code or if we are installing it. 
target_include_directories(Yanni PUBLIC
   # headers to include when building from source
   $<BUILD_INTERFACE:${Yanni_SOURCE_DIR}/include> 
   $<BUILD_INTERFACE:${Yanni_BINARY_DIR}/include> 
   "D:/vcpkg/installed/x64-windows/include/"
   # headers to include when installing  (implicitly prefixes with ${CMAKE_INSTALL_PREFIX}).
   $<INSTALL_INTERFACE:include> 
   )

# Install Yanni in CMAKE_INSTALL_PREFIX (defaults to /usr/local on linux). 
# To change the install location, run 
#   cmake -DCMAKE_INSTALL_PREFIX=<desired-install-path> ..

# install(...) specifies installation rules for the project. It can specify
# location of installed files on the system, user permissions, build
# configurations, etc. Here, we are only copying files.
# install(TARGETS ...) specifies rules for installing targets. 
# Here, we are taking a target or list of targets (Yanni) and telling CMake
# the following:
#   - put shared libraries associated with Yanni in ${Yanni_LIB_DEST}
#   - put static libraries associated with Yanni in ${Yanni_LIB_DEST}
#   - put include files associated with Yanni in ${Yanni_INCLUDE_DEST}
# We also need to specify the export that is associated with Yanni; an export 
# is just a list of targets to be installed.
# So we are associating Yanni with YanniTargets.
install(
    # targets to install
    TARGETS Yanni 
    # name of the CMake "export group" containing the targets we want to install
    EXPORT YanniTargets
    # Dynamic, static library and include destination locations after running
    # "make install"
    LIBRARY DESTINATION ${Yanni_LIB_DEST}
    ARCHIVE DESTINATION ${Yanni_LIB_DEST} 
    INCLUDES DESTINATION ${Yanni_INCLUDE_DEST}
    )

# We now need to install the export YanniTargets that we defined above. This
# is needed in order for another project to import Yanni using 
#   find_package(Yanni)
# find_package(Yanni) will look for Yanni-config.cmake to provide
# information about the targets contained in the project Yanni. Fortunately,
# this is specified in the export YanniTargets, so we will install this too.
# install(EXPORT ...) will install the information about an export. Here, we
# save it to a file {$Yanni_LIB_DEST}/YanniTargets.cmake and prepend 
# everything inside YanniTargets  with the namespace Yanni::.
install(
    # The export we want to save (matches name defined above containing the
    # install targets)
    EXPORT YanniTargets
    # CMake file in which to store the export's information
    FILE  YanniTargets.cmake
    # Namespace prepends all targets in the export (when we import later, we
    # will use Yanni::Yanni)
    NAMESPACE Yanni::
    # where to place the resulting file (here, we're putting it with the library)
    DESTINATION ${Yanni_LIB_DEST}
    )

# install(FILES ...) simply puts files in a certain place with certain
# properties. We're just copying them to the desired place here.
install(FILES ${Yanni_INC} DESTINATION ${Yanni_INCLUDE_DEST})

# Write a "version file" in case some wants to only load a particular version of
# Yanni @BUG this fails at the moment.
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    YanniConfigVersion.cmake
    VERSION ${Yanni_VERSION}
    COMPATIBILITY AnyNewerVersion
    )

# Copies the resulting CMake config files to the installed library directory
install(
    FILES 
        "cmake/Yanni-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/YanniConfigVersion.cmake"
    DESTINATION ${Yanni_LIB_DEST}
    )

target_link_libraries(YanniExec Yanni)
target_link_libraries(YanniExec yanniCudaLib)
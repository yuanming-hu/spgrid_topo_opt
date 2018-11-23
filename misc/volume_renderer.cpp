// Yuanming Hu, Dec 2017
// Based on the CUDA volume renderer demo

#include <experimental/filesystem>

#include <taichi/util.h>
#include "../topo_opt/fem_interface.h"
#include "../topo_opt/spgrid_topo_opt.h"
// OpenGL Graphics includes
#include <helper_gl.h>
#include <GLFW/glfw3.h>

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>

// CUDA utilities
#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <unistd.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>

GLFWwindow *window = nullptr;
int iDivUp(int a, int b) {
  return (a + b - 1) / b;
}
std::string channel_name = "density";

#include "volume_renderer.h"
#include <taichi/io/ply_writer.h>

TC_NAMESPACE_BEGIN

using Opt = SPGridTopologyOptimization3D;
using VolumeType = uchar;
// Size for the windows
uint width = 800, height = 800;

std::string mode = "";
class VisualizeDensity;
VisualizeDensity *visualizer = nullptr;
using BlockedGridUint8 =
    fem_interface::BlockedGrid<uint8, fem_interface::BlockSize<4, 4, 4>>;

struct VolumeRenderer {
  dim3 blockSize;
  dim3 gridSize;
  float3 viewRotation;
  float3 viewTranslation = make_float3(0.0, 0.0, -4.0f);
  float invViewMatrix[12];
  bool mirroring[3] = {false, false, false};
  real volume_rendering = 0.0_f;

  real density = 0.01f;
  real slice = 0.99999f;
  bool linearFiltering = false;

  uint pbo = 0;  // OpenGL pixel buffer object
  uint tex = 0;  // OpenGL texture object
  cudaGraphicsResource
      *cuda_pbo_resource;  // CUDA Graphics Resource (to transfer PBO)

  // Auto-Verification Code
  int fpsCount = 0;   // FPS count for averaging
  int fpsLimit = 10;  // FPS limit for sampling
  unsigned int frameCount = 0;
  std::vector<taichi::uint8> img_data;
  int channel = 0;

  VolumeRenderer() {
    blockSize = dim3(16, 16);
  }

  void computeFPS() {
    frameCount++;
    fpsCount++;
    if (fpsCount == fpsLimit) {
      real ifps = 1.f / (60);
      glfwSetWindowTitle(window,
                         fmt::format("TopoOpt Visualization - {:3.1f} fps - {}",
                                     ifps, channel_name)
                             .c_str());
      fpsCount = 0;
      fpsLimit = (int)std::max(1._f, ifps);
    }
  }

  void sample() {
    // map PBO to get CUDA device pointer
    uint *d_output;
    // map PBO to get CUDA device pointer
    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer(
        (void **)&d_output, &num_bytes, cuda_pbo_resource));
    // printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

    // clear image
    checkCudaErrors(cudaMemset(d_output, 0, width * height * 4));

    // call CUDA kernel, writing results to PBO
    render_kernel(gridSize, blockSize, d_output, width, height, density, slice,
                  volume_rendering);

    img_data.resize(width * height * 4);
    cudaMemcpy(&img_data[0], d_output, width * height * 4,
               cudaMemcpyDeviceToHost);

    getLastCudaError("kernel failed");

    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
  }

  // render image using CUDA
  void render() {
    // use OpenGL to build view matrix
    GLfloat modelView[16];
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glRotatef(-viewRotation.x, 1.0, 0.0, 0.0);
    glRotatef(-viewRotation.y, 0.0, 1.0, 0.0);
    glRotatef(-viewRotation.z, 0.0, 0.0, 1.0);
    glTranslatef(-viewTranslation.x, -viewTranslation.y, -viewTranslation.z);
    glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
    glPopMatrix();

    invViewMatrix[0] = modelView[0];
    invViewMatrix[1] = modelView[4];
    invViewMatrix[2] = modelView[8];
    invViewMatrix[3] = modelView[12];
    invViewMatrix[4] = modelView[1];
    invViewMatrix[5] = modelView[5];
    invViewMatrix[6] = modelView[9];
    invViewMatrix[7] = modelView[13];
    invViewMatrix[8] = modelView[2];
    invViewMatrix[9] = modelView[6];
    invViewMatrix[10] = modelView[10];
    invViewMatrix[11] = modelView[14];

    copyInvViewMatrix(invViewMatrix, sizeof(float4) * 3);

    sample();
  }

  void save_to_disk(const std::string &fn) {
    Array2D<Vector3> img(Vector2i(width, height));
    cudaDeviceSynchronize();
    for (int i = 0; i < (int)width; i++) {
      for (int j = 0; j < (int)height; j++) {
        for (int k = 0; k < 3; k++) {
          img[i][j][k] = (1 / 255.0_f) * img_data[(j * width + i) * 4 + k];
        }
      }
    }
    img.write_as_image(fn);
  }

  void display() {
    render();
    // display results
    glClear(GL_COLOR_BUFFER_BIT);

    // draw image from PBO
    glDisable(GL_DEPTH_TEST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // draw using texture
    // copy from pbo to texture
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA,
                    GL_UNSIGNED_BYTE, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    // draw textured quad
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0);
    glVertex2f(0, 0);
    glTexCoord2f(1, 0);
    glVertex2f(1, 0);
    glTexCoord2f(1, 1);
    glVertex2f(1, 1);
    glTexCoord2f(0, 1);
    glVertex2f(0, 1);
    glEnd();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glfwSwapBuffers(window);
    computeFPS();
  }

  ~VolumeRenderer() {
    if (pbo) {
      cudaGraphicsUnregisterResource(cuda_pbo_resource);
      glDeleteBuffers(1, &pbo);
      glDeleteTextures(1, &tex);
    }
    // Calling cudaProfilerStop causes all profile data to be
    // flushed before the application exits
    checkCudaErrors(cudaProfilerStop());
  }

  void initPixelBuffer() {
    if (pbo) {
      // unregister this buffer object from CUDA C
      checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

      // delete old buffer
      glDeleteBuffers(1, &pbo);
      glDeleteTextures(1, &tex);
    }

    // create pixel buffer object for display
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB,
                 width * height * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    // register this buffer object with CUDA
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(
        &cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

    // create texture for display
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
  }

  void key_down(int key, int modes);

  void reshape(int w, int h) {
    reset_render_buffer(w, h);
    initPixelBuffer();
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));
  }

  void reset_canvas() {
    reset_render_buffer(width, height);
  }

  void toggle_mirroring(int k) {
    mirroring[k] ^= 1;
    set_mirroring(mirroring);
    TC_TRACE("mirroring {} {} {}", mirroring[0], mirroring[1], mirroring[2]);
  }

} renderer;

int ox, oy;
int mx, my;
int buttonState = -1;

void motion(int x, int y) {
  float dx, dy;
  dx = (float)(x - ox);
  dy = (float)(y - oy);

  if (buttonState == GLFW_MOUSE_BUTTON_RIGHT) {
    // right = zoom
    renderer.viewTranslation.z -= dy / 100.0f;
  } else if (buttonState == GLFW_MOUSE_BUTTON_MIDDLE) {
    // middle = translate
    renderer.viewTranslation.x += dx / 100.0f;
    renderer.viewTranslation.y -= dy / 100.0f;
  } else if (buttonState == GLFW_MOUSE_BUTTON_LEFT) {
    // left = rotate
    renderer.viewRotation.x -= dy / 5.0f;
    renderer.viewRotation.y += dx / 5.0f;
  }
  ox = x;
  oy = y;
}

void reshape(GLFWwindow *window, int w, int h) {
  width = w;
  height = h;
  renderer.reshape(w, h);

  glViewport(0, 0, w, h);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 1, 0, 1, 0, 1);
  getLastCudaError("reshape failed");
}

void initGL() {
  glfwSetErrorCallback([](int error, const char *description) {
    fprintf(stderr, "Error: %s\n", description);
  });
  if (!glfwInit()) {
    TC_ERROR("GLFW initialization failed.");
  }
  // TODO: more modern version?
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  window = glfwCreateWindow(width, height, "TopoOpt Visualizer", NULL, NULL);
  if (!window) {
    TC_ERROR("GLFW Window creation failed!");
  }

  glfwMakeContextCurrent(window);

  if (!isGLVersionSupported(2, 0) ||
      !areGLExtensionsSupported("GL_ARB_pixel_buffer_object")) {
    printf("Required OpenGL extensions are missing.");
    exit(EXIT_SUCCESS);
  }
}

class VisualizeDensity : public Task {
 public:
  std::string suffix;
  std::vector<std::string> file_names;
  std::vector<uchar> h_volume;
  cudaExtent volumeSize;
  int current_file_id;
  bool volume_initialized;
  Vector3i res;
  Opt opt;
  int channel = 0;
  std::string folder_name;
  Vector3i min_coord_preset;

  void load_residual(const std::string &file_name) {
    using namespace fem_interface;
    TC_INFO("Loading {}", file_name);
    using block_size_ = BlockSize<4, 4, 4>;
    using FloatScalarGrid = BlockedGrid<float, block_size_>;
    FloatScalarGrid residual;

    read_from_binary_file(residual, file_name);
    TC_TRACE("File loaded.");
    Vector3i min_coord(1000000000);
    Vector3i max_coord(0);
    using scalar_block_size = FloatScalarGrid::block_size;
    auto block_size = Vector3i(scalar_block_size::x, scalar_block_size::y,
                               scalar_block_size::z);
    /*
    for (auto block : residual.blocks) {
      for (auto ind : RegionND<3>(Vector3i(0), block_size)) {
        for (int i = 0; i < 3; i++) {
          if (block.get(ind.i, ind.j, ind.k) > 0) {
            min_coord[i] = std::min(
                min_coord[i], block.base_coordinates[i] + ind.get_ipos()[i]);
            max_coord[i] =
                std::max(max_coord[i],
                         block.base_coordinates[i] + ind.get_ipos()[i] + 1);
          }
        }
      }
    }
    */
    max_coord = Vector3i(1032, 1032, 220);
    min_coord = Vector3i(8);
    // max_coord += block_size;
    create_volume(max_coord - min_coord);
    TC_P(res);
    std::fill(h_volume.begin(), h_volume.end(), 0);
    int outside = 0;
    for (auto block : residual.blocks) {
      auto base_coord = Vector3i::from_array(block.base_coordinates);
      for (auto ind : RegionND<3>(Vector3i(0), block_size)) {
        auto coord = base_coord + ind.get_ipos() - min_coord;
        auto data = block.get(ind.i, ind.j, ind.k);
        if (Vector3i(0) <= coord && coord < res) {
          if (data == 0) {
            h_volume[coord.x + coord.y * volumeSize.width +
                     coord.z * volumeSize.width * volumeSize.height] = 0;
          } else {
            h_volume[coord.x + coord.y * volumeSize.width +
                     coord.z * volumeSize.width * volumeSize.height] =
                uchar(data * 254) + 1;
          }
        } else {
          outside += 1;
        }
      }
    }
    for (int j = 0; j < res.z; j++) {
      for (int i = 0; i < res.x; i++) {
        for (int k = 0; k < 0; k++) {
          h_volume[i + k * volumeSize.width +
                   j * volumeSize.width * volumeSize.height] = 255;
        }
      }
    }
    if (outside) {
      TC_WARN("{} outside voxels detected", outside);
    }
    refresh_volume();
  }

  void load_fem(const std::string &file_name) {
    using namespace fem_interface;
    TC_INFO("Loading {}", file_name);
    FEMInputs param;
    read_from_binary_file(param, file_name);
    TC_TRACE("File loaded.");
    Vector3i min_coord(1000000000);
    Vector3i max_coord(0);
    using scalar_block_size = FEMInputs::ScalarGrid::block_size;
    auto block_size = Vector3i(scalar_block_size::x, scalar_block_size::y,
                               scalar_block_size::z);
    for (auto block : param.density.blocks) {
      for (auto ind : RegionND<3>(Vector3i(0), block_size)) {
        for (int i = 0; i < 3; i++) {
          if (block.get(ind.i, ind.j, ind.k) > 0) {
            min_coord[i] = std::min(
                min_coord[i], block.base_coordinates[i] + ind.get_ipos()[i]);
            max_coord[i] =
                std::max(max_coord[i],
                         block.base_coordinates[i] + ind.get_ipos()[i] + 1);
          }
        }
      }
    }
    // max_coord += block_size;
    //#max_coord = Vector3i(308);
    // min_coord = Vector3i(8);
    // max_coord = Vector3i(1032, 1032, 626);
    // min_coord = Vector3i(8, 8, 220);
    // max_coord += block_size;
    TC_P(res);
    create_volume(max_coord - min_coord);
    std::fill(h_volume.begin(), h_volume.end(), 0);
    int outside = 0;
    real sum = 0;
    bool reverse_mapping = param.penalty() == 0;
    for (auto block : param.density.blocks) {
      auto base_coord = Vector3i::from_array(block.base_coordinates);
      for (auto ind : RegionND<3>(Vector3i(0), block_size)) {
        auto coord = base_coord + ind.get_ipos() - min_coord;
        auto data = block.get(ind.i, ind.j, ind.k);
        sum += data;
        if (reverse_mapping) {
          data = std::pow(data - 1e-6_f, 0.33333333_f) * (1 - 1e-6_f);
        }
#if defined(WIREFRAME)
        // NOTE: for visualizing SPGrid narrow band!
        data = 0;
#endif
        if (Vector3i(0) <= coord && coord < res) {
          if (data == 0) {
            h_volume[coord.x + coord.y * volumeSize.width +
                     coord.z * volumeSize.width * volumeSize.height] = 0;
          } else {
#if defined(WIREFRAME)
            h_volume[coord.x + coord.y * volumeSize.width +
                     coord.z * volumeSize.width * volumeSize.height] =
                uchar(data * 240) + 1;
#else
            h_volume[coord.x + coord.y * volumeSize.width +
                     coord.z * volumeSize.width * volumeSize.height] =
                uchar(data * 254) + 1;
#endif
          }
        } else {
          outside += 1;
        }
      }
#if defined(WIREFRAME)
      // Draw a wireframe for the block
      Vector3i ends[2];
      ends[0] = Vector3i(0);
      ends[1] = block_size * Vector3i(1, 1, 2) + Vector3i(1);
      base_coord.z = base_coord.z / 8 * 8;
      for (auto ind : RegionND<3>(ends[0], ends[1])) {
        int count = 0;
        for (int i = 0; i < 3; i++) {
          count += ind.get_ipos()[i] == ends[0][i];
          count += ind.get_ipos()[i] == ends[1][i] - 1;
        }
        if (count >= 2) {  // set to 2 for blocks
          auto coord = base_coord + ind.get_ipos() - min_coord;
          if (Vector3i(0) <= coord && coord < res) {
            h_volume[coord.x + coord.y * volumeSize.width +
                     coord.z * volumeSize.width * volumeSize.height] = 255;
          }
        }
      }
#endif
    }
    TC_P(sum);

    for (int j = 0; j < res.z; j++) {
      for (int i = 0; i < res.x; i++) {
        for (int k = 0; k < 0; k++) {
          h_volume[i + k * volumeSize.width +
                   j * volumeSize.width * volumeSize.height] = 255;
        }
      }
    }
    if (outside) {
      TC_WARN("{} outside voxels detected", outside);
    }
    refresh_volume();
  }

  void load_fem_quick(const std::string &file_name) {
    using namespace fem_interface;
    TC_INFO("Loading {}", file_name);
    BlockedGridUint8 param;
    read_from_binary_file(param, file_name);
    TC_TRACE("File loaded.");
    Vector3i min_coord(1000000000);
    Vector3i max_coord(0);
    using scalar_block_size = FEMInputs::ScalarGrid::block_size;
    auto block_size = Vector3i(scalar_block_size::x, scalar_block_size::y,
                               scalar_block_size::z);
    for (auto block : param.blocks) {
      for (auto ind : RegionND<3>(Vector3i(0), block_size)) {
        for (int i = 0; i < 3; i++) {
          if (block.get(ind.i, ind.j, ind.k) > 0) {
            min_coord[i] = std::min(
                min_coord[i], block.base_coordinates[i] + ind.get_ipos()[i]);
            max_coord[i] =
                std::max(max_coord[i],
                         block.base_coordinates[i] + ind.get_ipos()[i] + 1);
          }
        }
      }
    }
    max_coord += block_size;
    // TC_P(max_coord);
    // TC_P(min_coord);
    max_coord = Vector3i(1032, 1032, 626);
    min_coord = Vector3i(8, 8, 220);
    create_volume(max_coord - min_coord);
    // max_coord += block_size;
    // create_volume(max_coord - min_coord);
    std::fill(h_volume.begin(), h_volume.end(), 0);
    int outside = 0;
    for (auto block : param.blocks) {
      auto base_coord = Vector3i::from_array(block.base_coordinates);
      for (auto ind : RegionND<3>(Vector3i(0), block_size)) {
        auto coord = base_coord + ind.get_ipos() - min_coord;
        auto data = block.get(ind.i, ind.j, ind.k);
        if (Vector3i(0) <= coord && coord < res) {
          h_volume[coord.x + coord.y * volumeSize.width +
                   coord.z * volumeSize.width * volumeSize.height] = data;
        } else {
          outside += 1;
        }
      }
    }
    if (outside) {
      TC_WARN("{} outside voxels detected", outside);
    }
    refresh_volume();
  }
  void create_volume(Vector3i res) {
    this->res = res;
    if (!volume_initialized) {
      TC_P(res);
      Vector3f fres = res.template cast<float32>() / float32(res.max());
      set_box_size(&fres[0]);
      set_box_res(&res[0]);
    }
    volumeSize.width = std::size_t(res.x);
    volumeSize.height = std::size_t(res.y);
    volumeSize.depth = std::size_t(res.z);
    h_volume = std::vector<uchar>(
        volumeSize.width * volumeSize.height * volumeSize.depth, 0);
    set_mirroring(renderer.mirroring);
  }

  TC_FORCE_INLINE real get_voxel(Vector3i coord, real outside) {
    if (Vector3i(0) <= coord &&
        coord <
            Vector3i(volumeSize.width, volumeSize.height, volumeSize.depth)) {
      auto val = h_volume[coord.x + coord.y * volumeSize.width +
                          coord.z * volumeSize.width * volumeSize.height] *
                 (1 / 255.0_f);
      return val;
    } else {
      return outside;
    }
  }

  TC_FORCE_INLINE real get_big_voxel(Vector3i coord, int scale, real outside) {
    real sum = 0;
    for (int i = 0; i < scale; i++) {
      for (int j = 0; j < scale; j++) {
        for (int k = 0; k < scale; k++) {
          sum += get_voxel(coord * scale + Vector3i(i, j, k), outside);
        }
      }
    }
    return sum / pow<3>(scale);
  }

  TC_FORCE_INLINE bool set_voxel(Vector3i coord, real val) {
    if (Vector3i(0) <= coord && coord < res) {
      val = clamp(val, 0.0_f, 1.0_f);
      h_volume[coord.x + coord.y * volumeSize.width +
               coord.z * volumeSize.width * volumeSize.height] =
          uchar(val * 255);
      return true;
    } else {
      return false;
    }
  }

  void load_channel() {
    int outside = 0;
    auto blocks = opt.grid->fat_page_map->Get_Blocks();
    using Array = decltype(opt.grid->density());

    auto load = [&](Array data, const std::string &name) {
      channel_name = name;
      fmt::print("\nShowing channel {}\n\n", name);
      auto flags = opt.grid->flags();
      Vector3i origin = opt.container_bounds[0];
      real maximum = -1e30_f;
      real minimum = 1e30_f;

      for (uint i = 0; i < blocks.second; i++) {
        auto offset = blocks.first[i];
        Vector3i base_coord(Opt::ChannelMask::LinearToCoord(offset));
        for (auto ind : RegionND<3>(Vector3i(0), opt.block_size)) {
          auto i = base_coord + ind.get_ipos();
          if (flags(i).get_inside_container()) {
            real value = data(i);
            maximum = std::max(maximum, value);
            minimum = std::min(minimum, value);
          }
        }
      }
      TC_P(minimum);
      TC_P(maximum);
      bool use_abs;
      if (minimum < 0) {
        use_abs = true;
      }

      for (uint i = 0; i < blocks.second; i++) {
        auto offset = blocks.first[i];
        Vector3i base_coord(Opt::ChannelMask::LinearToCoord(offset));
        for (auto ind : RegionND<3>(Vector3i(0), opt.block_size)) {
          auto i = base_coord + ind.get_ipos();
          if (flags(i).get_inside_container()) {
            real value = data(i);
            if (use_abs) {
              value = std::abs(value);
            }
            bool ret = set_voxel(i - origin, value / maximum);
            if (!ret) {
              outside += 1;
            }
          }
        }
      }
      if (outside) {
        TC_WARN("{} outside voxels", outside);
      }
    };

    if (channel == 0) {
      load(opt.grid->density(), std::string("Density"));
    } else if (channel == 1) {
      load(opt.grid->dc0(), std::string("Smoothed sensitivity"));
    } else if (channel == 2) {
      load(opt.grid->dc1(), std::string("Raw sensitivity"));
    } else if (channel == 3) {
      TC_NOT_IMPLEMENTED;
      // load(opt.grid->u0(), std::string("U_x"));
    } else if (channel == 4) {
      TC_NOT_IMPLEMENTED;
      // load(opt.grid->u1(), std::string("U_y"));
    } else if (channel == 5) {
      TC_NOT_IMPLEMENTED;
      // load(opt.grid->u2(), std::string("U_z"));
    } else if (channel == 6) {
      load(opt.grid->residual(), std::string("Residual"));
    }
    refresh_volume();
  }

  void save_as_obj(real slice_start = 0,
                   real slice_end = 1,
                   int color_start = 0) {
    using Vectori = TVector<int, 3>;
    using Vector = TVector<real, 3>;
    for (int color = 0; color < 2; color++) {
      PLYWriter writer(fmt::format("/tmp/{}.ply", color + color_start));
      real scale = res.x;
      int downsample = 2;
      auto printing_res = res / downsample;
      TC_P(printing_res);
      real threshold = renderer.density;
      Vectori offsets[] = {Vectori(1, 0, 0), Vectori(-1, 0, 0),
                           Vectori(0, 1, 0), Vectori(0, -1, 0),
                           Vectori(0, 0, 1), Vectori(0, 0, -1)};
      Vector dx[] = {Vector(-1, -1, -1), Vector(-1, -1, 1), Vector(-1, 1, -1),
                     Vector(-1, 1, 1),   Vector(1, -1, -1), Vector(1, -1, 1),
                     Vector(1, 1, -1),   Vector(1, 1, 1)};
      const int vid[6][4] = {{4, 5, 7, 6}, {2, 3, 1, 0}, {2, 6, 7, 3},
                             {0, 1, 5, 4}, {1, 3, 7, 5}, {0, 4, 6, 2}};
      auto flip = [&](real x) -> real {
        if (color) {
          return x;
        } else {
          return 1 - x;
        }
      };

      int bound0 = int(slice_start * printing_res.y);
      int bound1 = int(slice_end * printing_res.y);
      auto V = [&](Vector3i v, int downsample, real outside) -> real {
        if (v.y < bound0 || v.y >= bound1) {
          return outside;
        }
        return get_big_voxel(v, downsample, color);
      };
      for (auto &ind : RegionND<3>(Vectori(-2), printing_res + Vectori(3))) {
        if (flip(V((ind).get_ipos(), downsample, color)) > flip(threshold)) {
          continue;
        }
        for (int k = 0; k < 6; k++) {
          auto o = offsets[k];
          auto nind = ind.get_ipos() + o;
          if (flip(V((nind), downsample, color)) > flip(threshold)) {
            Vector base_pos = ind.get_pos() - Vector(scale / 2.0_f);
            std::vector<Vector> vertices;
            vertices.push_back((base_pos + 0.5_f * dx[vid[k][0]]) *
                               (1.0_f / scale));
            vertices.push_back((base_pos + 0.5_f * dx[vid[k][1]]) *
                               (1.0_f / scale));
            vertices.push_back((base_pos + 0.5_f * dx[vid[k][2]]) *
                               (1.0_f / scale));
            writer.add_face(vertices);
            vertices.clear();
            vertices.push_back((base_pos + 0.5_f * dx[vid[k][0]]) *
                               (1.0_f / scale));
            vertices.push_back((base_pos + 0.5_f * dx[vid[k][2]]) *
                               (1.0_f / scale));
            vertices.push_back((base_pos + 0.5_f * dx[vid[k][3]]) *
                               (1.0_f / scale));
            writer.add_face(vertices);
          }
        }
      }
      TC_INFO("Color {} done.", color);
    }
  }

  void refresh_volume() {
    if (!volume_initialized) {
      initCuda(&h_volume[0], volumeSize);
      volume_initialized = true;
    } else {
      update_volume(&h_volume[0], volumeSize);
    }
  }

  void load_snapshot(const std::string &file_name) {
    using namespace fem_interface;
    TC_INFO("Loading snapshot {}", file_name);
    opt.general_action(
        Dict().set("name", "load_state").set("filename", file_name));
    TC_TRACE("Snapshot loaded.");
    TC_P(opt.get_container_size());

    create_volume(opt.get_container_size() + Vector3i(1));

    load_channel();
  }

  void load(std::string fn) {
    if (mode == "snapshots") {
      load_snapshot(fn);
    } else if (mode == "fem") {
      load_fem(fn);
    } else if (mode == "fem_quick") {
      load_fem_quick(fn);
    } else if (mode == "residual") {
      load_residual(fn);
    } else {
      TC_ERROR("Unknown mode: {}", mode);
    }
  }

  std::string run(const std::vector<std::string> &parameters) override;

  void reload(int offset) {
    current_file_id += offset + (int)file_names.size();
    current_file_id %= (int)file_names.size();
    load(file_names[current_file_id]);
  }

  using Script = std::function<void(int, float)>;

  void generate_video(const Script &script, int count);
};

std::string VisualizeDensity::run(const std::vector<std::string> &parameters_) {
  auto parameters = parameters_;
  visualizer = this;
  volume_initialized = false;
#if defined(__linux__)
  setenv("DISPLAY", ":0", 0);
#endif
  current_file_id = 0;
  // First initialize OpenGL context, so we can properly set the GL for
  // CUDA. This is necessary in order to achieve optimal performance with
  // OpenGL/CUDA interop.
  initGL();

  // Use device with highest Gflops/s
  cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

  std::string file_name = ".";
  if (parameters.empty()) {
    parameters.push_back(".");
  }
  if (parameters.size() == 1) {
    parameters.push_back(parameters[0]);
    parameters[0] = "fem";
    TC_INFO("Mode set to 'fem'");
  }
  mode = parameters[0];
  if (mode == "s") {
    mode = "snapshots";
  }
  if (mode == "r") {
    mode = "residual";
  }
  if (mode == "q") {
    mode = "fem_quick";
  }
  TC_ASSERT_INFO(
      mode == "fem" || mode == "fem_quick" || mode == "snapshots" ||
          mode == "residual",
      "The first parameter to visualize_density should be [mode=fem, "
      "snapshots]")
  if (!parameters.empty()) {
    file_name = parameters[1];
  }
  if (ends_with(file_name, ".tcb") || ends_with(file_name, "tcb" + suffix)) {
    folder_name = ".";
    file_names.push_back(file_name);
  } else {
    if (std::experimental::filesystem::exists(file_name + "/" + mode)) {
      TC_INFO("Found an '{}' folder. Diving into..", mode);
      file_name += "/" + mode;
    }
    TC_INFO("Scanning directory {} ...", file_name);
    folder_name = file_name;
    for (auto &p :
         std::experimental::filesystem::directory_iterator(file_name)) {
      std::string tcb_file_name = p.path().filename();
      if (ends_with(tcb_file_name, ".tcb") ||
          ends_with(tcb_file_name, "tcb" + suffix)) {
        file_names.push_back(file_name + "/" + tcb_file_name);
      }
    }
  }
  std::sort(std::begin(file_names), std::end(file_names));
  TC_INFO("  {} tcb files found.", file_names.size());
  TC_ASSERT_INFO(file_names.size() > 0,
                 "There should be at least one tcb file.");
  current_file_id = file_names.size() - 1;
  load(file_names[current_file_id]);

  glfwSetKeyCallback(window, [](GLFWwindow *window, int key, int scancode,
                                int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
      glfwSetWindowShouldClose(window, true);
    if (action == GLFW_RELEASE)
      return;
    if (key == GLFW_KEY_C) {
      if (mods & GLFW_MOD_SHIFT) {
        visualizer->channel -= 1;
      } else {
        visualizer->channel += 1;
      }
      visualizer->channel += 7;
      visualizer->channel %= 7;
      visualizer->load_channel();
    }

    renderer.key_down(key, mods);
  });

  glfwSetMouseButtonCallback(
      window, [](GLFWwindow *window, int button, int action, int mods) {
        reset_render_buffer(width, height);
        if (action == GLFW_PRESS) {
          ox = mx;
          oy = my;
          buttonState = button;
        } else if (action == GLFW_RELEASE) {
          buttonState = -1;
        }
      });

  glfwSetCursorPosCallback(window,
                           [](GLFWwindow *window, double xpos, double ypos) {
                             mx = xpos;
                             my = ypos;
                           });

  glfwSetWindowSizeCallback(window, reshape);

  glfwSetScrollCallback(window,
                        [](GLFWwindow *window, double xoffset, double yoffset) {
                          reset_render_buffer(width, height);
                          renderer.viewTranslation.z += yoffset * 0.2_f;
                        });

  reshape(window, width, height);

  /*
  //save_as_obj();
  renderer.density = 0.3_f;
  save_as_obj(0.00, 0.4, 0);
  save_as_obj(0.4, 1, 2);
  //save_as_obj(0.67, 1, 4);
  exit(0);
  */

  while (!glfwWindowShouldClose(window)) {
    glViewport(0, 0, width, height);
    renderer.render();
    renderer.display();
    glfwSwapBuffers(window);
    glfwPollEvents();
    if (buttonState != -1)
      motion(mx, my);
  }
  glfwDestroyWindow(window);
  glfwTerminate();
  exit(EXIT_SUCCESS);
  return "";
}

void VolumeRenderer::key_down(int key, int modes) {
  switch (key) {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, true);
      break;

    case GLFW_KEY_F:
      linearFiltering = !linearFiltering;
      setTextureFilterMode(linearFiltering);
      break;

    case GLFW_KEY_J:
      density -= 0.005f;
      break;

    case GLFW_KEY_K:
      density += 0.005f;
      break;

    case GLFW_KEY_H:
      visualizer->reload(-1);
      break;

    case GLFW_KEY_L:
      visualizer->reload(1);
      break;

    case GLFW_KEY_X:
      slice += 0.01;
      break;

    case GLFW_KEY_Z:
      slice -= 0.01;
      break;

    case GLFW_KEY_S:
      if (modes & GLFW_MOD_SHIFT) {
        visualizer->save_as_obj();
      } else {
        save_to_disk("/tmp/a.png");
      }
      break;

    case GLFW_KEY_R:
      volume_rendering = volume_rendering * 0.7_f - 1e-4_f;
      break;

    case GLFW_KEY_T:
      volume_rendering = volume_rendering / 0.7_f + 1e-4_f;
      break;

    case GLFW_KEY_1:
      renderer.toggle_mirroring(0);
      break;

    case GLFW_KEY_2:
      renderer.toggle_mirroring(1);
      break;

    case GLFW_KEY_3:
      renderer.toggle_mirroring(2);
      break;

    case GLFW_KEY_Q:
      renderer.viewRotation.z += 30;
      break;

    case GLFW_KEY_V:
      if (modes & GLFW_MOD_SHIFT)
        visualizer->generate_video(
            [&](int i, real r) { visualizer->load(visualizer->file_names[i]); },
            visualizer->file_names.size());
      break;

    case GLFW_KEY_B:
      if (modes & GLFW_MOD_SHIFT)
        visualizer->generate_video([&](int i, real r) { slice = r; }, 200);
      break;

    case GLFW_KEY_N:
      if (modes & GLFW_MOD_SHIFT) {
        auto initialRotation = viewRotation.y;
        visualizer->generate_video(
            [&](int i, real r) {
              viewRotation.y =
                  initialRotation + 360 * (1 - std::cos(r * pi)) * 0.5_f;
            },
            240);
      }
      break;

    case GLFW_KEY_M:
      if (modes & GLFW_MOD_SHIFT) {
        auto initialRotation = viewRotation.x;
        visualizer->generate_video(
            [&](int i, real r) {
              viewRotation.x =
                  initialRotation + 360 * (1 - std::cos(r * pi)) * 0.5_f;
            },
            240);
      }
      break;

    case GLFW_KEY_P:
      if (modes & GLFW_MOD_SHIFT) {
        auto initialRotation = viewRotation.y;
        visualizer->generate_video(
            [&](int i, real r) {
              viewRotation.y =
                  initialRotation - 30 * (1 - std::cos(r * pi)) * 0.5_f;
            },
            120);
      }
      break;

    default:
      break;
  }
  slice = fract(slice);
  density = clamp(density, 0.0_f, 1.0_f);
  volume_rendering = std::max(volume_rendering, 0.0_f);

  fmt::print(
      "density = {:.2}, file_name = {}, slice position = {}, volume_rendering "
      "= {}\n",
      density, visualizer->file_names[visualizer->current_file_id], slice,
      volume_rendering);

  reset_render_buffer(width, height);
}

void VisualizeDensity::generate_video(const VisualizeDensity::Script &script,
                                      int count) {
  std::string folder = fmt::format("{}/video/", folder_name);
  std::experimental::filesystem::create_directory(folder);
  for (int i = 0; i < count; i++) {
    reset_render_buffer(width, height);
    script(i, 1.0_f * i / count);
    for (int k = 0; k < 100; k++) {
      renderer.render();
    }
    renderer.save_to_disk(fmt::format("{}/{:04d}.png", folder, i));
  }
  trash(std::system(fmt::format("cd {} && ti video", folder).c_str()));
}

auto vd = [](const std::vector<std::string> &parameters) {
  VisualizeDensity vd;
  vd.suffix = ".zip";
  vd.run(parameters);
};

TC_REGISTER_TASK(vd);

auto cd = [](const std::vector<std::string> &parameters) {
  for (auto &fn : parameters) {
    using namespace fem_interface;
    auto dir = std::string("/tmp/topopt/");
    std::error_code ec;
    std::experimental::filesystem::create_directories(dir, ec);
    TC_INFO("Loading {}", fn);
    FEMInputs param;
    read_from_binary_file(param, fn);
    TC_TRACE("File loaded.");
    using scalar_block_size = FEMInputs::ScalarGrid::block_size;
    auto block_size = Vector3i(scalar_block_size::x, scalar_block_size::y,
                               scalar_block_size::z);
    BlockedGridUint8 dataChar;
    for (auto block : param.density.blocks) {
      auto base_coord = Vector3i::from_array(block.base_coordinates);
      BlockedGridUint8::Block charBlock;
      for (int i = 0; i < 3; i++) {
        charBlock.base_coordinates[i] = block.base_coordinates[i];
      }
      for (auto ind : RegionND<3>(Vector3i(0), block_size)) {
        auto data = block.get(ind.i, ind.j, ind.k);
        char ch;
        if (data == 0) {
          ch = 0;
        } else {
          ch = uchar(data * 254) + 1;
        }
        charBlock.get(ind.i, ind.j, ind.k) = ch;
      }
      dataChar.blocks.push_back(charBlock);
    }
    std::string suffix = "";
    if (!ends_with(fn, ".zip")) {
      suffix = ".zip";
    }
    write_to_binary_file(dataChar, dir + "small" + fn + suffix);
  }

};

TC_IMPLEMENTATION(Task, VisualizeDensity, "visualize_density");
TC_REGISTER_TASK(cd);

TC_NAMESPACE_END

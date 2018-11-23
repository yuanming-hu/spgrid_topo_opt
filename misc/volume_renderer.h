void set_box_size(float *size);
void setTextureFilterMode(bool bLinearFilter);
void reset_render_buffer(int width, int height);
void initCuda(void *h_volume, cudaExtent volumeSize);
void update_volume(void *h_volume, cudaExtent volumeSize);
void freeCudaBuffers();
void render_kernel(dim3 gridSize,
                   dim3 blockSize,
                   uint *d_output,
                   uint imageW,
                   uint imageH,
                   float density,
                   float slice_position,
                   float volume_rendering);
void copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix);
void set_box_res(int *res);
void set_mirroring(bool *new_mirroring);

// #define WIREFRAME

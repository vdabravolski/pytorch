#version 310 es
layout(std430) buffer;
layout(rgba32f, binding=0) writeonly uniform mediump image3D uOutput;
layout(binding=2) readonly buffer kernel{
    vec4 data[];
} uKernel;

layout(location = 3) uniform int uKWxKH;
layout(location = 4) uniform int uC_4;

layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID) * ivec3(4, 1, 1);
  int bufferIdx = pos.x*uKWxKH + 4*pos.y*uC_4*uKWxKH + 4*pos.z;
  vec4 v0 = uKernel.data[bufferIdx+0];
  vec4 v1 = uKernel.data[bufferIdx+1];
  vec4 v2 = uKernel.data[bufferIdx+2];
  vec4 v3 = uKernel.data[bufferIdx+3];

  imageStore(uOutput, ivec3(pos.x+0, pos.y, pos.z), v0);
  imageStore(uOutput, ivec3(pos.x+1, pos.y, pos.z), v1);
  imageStore(uOutput, ivec3(pos.x+2, pos.y, pos.z), v2);
  imageStore(uOutput, ivec3(pos.x+3, pos.y, pos.z), v3);
}

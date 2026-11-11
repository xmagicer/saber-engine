#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <chrono>
#include <vector>

static void hip_check(hipError_t s, const char *what) {
    if (s != hipSuccess) throw std::runtime_error(std::string(what)+": "+hipGetErrorName(s));
}
static void rccl_check(ncclResult_t s, const char *what) {
    if (s != ncclSuccess) throw std::runtime_error(std::string(what)+": "+ncclGetErrorString(s));
}
#define H(x) hip_check((x), #x)
#define N(x) rccl_check((x), #x)

__host__ __device__ static uint16_t input_bits(size_t i, int rank, int seed) {
    return static_cast<uint16_t>((rank == 0 ? 0x3f00 : 0x3c00) + ((i*7+seed*11)%257));
}
__global__ void fill_bf16(uint16_t *p, size_t n, int rank, int seed) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) p[i]=input_bits(i,rank,seed);
}
__global__ void fill_bytes(uint8_t *p, size_t n, int rank, int seed) {
    size_t i=blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) p[i]=static_cast<uint8_t>((i*13+rank*37+seed)&255);
}
static uint16_t expected_sum(size_t i, int seed) {
    uint32_t a=uint32_t(input_bits(i,0,seed))<<16, b=uint32_t(input_bits(i,1,seed))<<16;
    float x,y;std::memcpy(&x,&a,4);std::memcpy(&y,&b,4);float z=x+y;
    uint32_t bits;std::memcpy(&bits,&z,4);
    return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);
}

int main(int argc,char **argv) {
    if (argc!=3) {std::fprintf(stderr,"usage: probe RANK UNIQUE_ID_FILE\n");return 2;}
    int rank=std::stoi(argv[1]);std::string id_file=argv[2];
    try {
        int devices=0,peer=0,version=0;H(hipGetDeviceCount(&devices));
        if (devices!=2 || rank<0 || rank>1) throw std::runtime_error("Expected two devices and rank 0/1");
        H(hipSetDevice(rank));H(hipDeviceCanAccessPeer(&peer,rank,1-rank));N(ncclGetVersion(&version));
        std::printf("{\"rank\":%d,\"physical_peer_access\":%d,\"rccl_version\":%d}\n",rank,peer,version);std::fflush(stdout);
        ncclUniqueId id;
        if (rank==0) {
            N(ncclGetUniqueId(&id));std::ofstream out(id_file,std::ios::binary|std::ios::trunc);
            out.write(reinterpret_cast<const char*>(&id),sizeof(id));out.close();
        } else {
            bool found=false;
            for (int i=0;i<300;++i) {
                std::ifstream in(id_file,std::ios::binary|std::ios::ate);
                if (in && in.tellg()==static_cast<std::streampos>(sizeof(id))) {
                    in.seekg(0);in.read(reinterpret_cast<char*>(&id),sizeof(id));found=true;break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (!found) throw std::runtime_error("Unique ID exchange timed out");
        }
        ncclComm_t comm;ncclConfig_t config=NCCL_CONFIG_INITIALIZER;config.graphUsageMode=2;config.blocking=1;N(ncclCommInitRankConfig(&comm,2,id,rank,&config));
        hipStream_t stream;H(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        for (size_t n:{size_t(5120),size_t(131072),size_t(4194304)}) {
            uint16_t *send=nullptr,*recv=nullptr;H(hipMalloc(&send,n*2));H(hipMalloc(&recv,n*2));
            std::vector<uint16_t> host(n);
            for (int graph_mode=0;graph_mode<2;++graph_mode) {
                hipGraph_t graph=nullptr;hipGraphExec_t exec=nullptr;
                hipLaunchKernelGGL(fill_bf16,dim3((n+255)/256),dim3(256),0,stream,send,n,rank,1);
                N(ncclAllReduce(send,recv,n,ncclBfloat16,ncclSum,comm,stream));H(hipStreamSynchronize(stream));
                if (graph_mode) {
                    H(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
                    for (int call=0;call<16;++call) {
                        N(ncclAllReduce(send,recv,n,ncclBfloat16,ncclSum,comm,stream));
                        hipLaunchKernelGGL(fill_bytes,dim3(1),dim3(256),0,stream,reinterpret_cast<uint8_t*>(recv),size_t(0),rank,0);
                    }
                    H(hipStreamEndCapture(stream,&graph));H(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
                    size_t nodes=0;H(hipGraphGetNodes(graph,nullptr,&nodes));
                    std::vector<hipGraphNode_t> ns(nodes);H(hipGraphGetNodes(graph,ns.data(),&nodes));
                    int types[20]={};for(auto node:ns){hipGraphNodeType t;H(hipGraphNodeGetType(node,&t));if(int(t)<20)types[int(t)]++;}
                    std::printf("NODE_TYPES rank=%d nodes=%zu",rank,nodes);for(int t=0;t<20;++t)if(types[t])std::printf(" type%d=%d",t,types[t]);std::puts("");
                    std::printf("{\"rank\":%d,\"op\":\"allreduce\",\"elements\":%zu,\"graph_nodes\":%zu}\n",rank,n,nodes);std::fflush(stdout);
                }
                double elapsed_ms=0;
                for (int repetition=0;repetition<20;++repetition) {
                    int seed=repetition+17;
                    hipLaunchKernelGGL(fill_bf16,dim3((n+255)/256),dim3(256),0,stream,send,n,rank,seed);
                    H(hipGetLastError());H(hipStreamSynchronize(stream));
                    auto start=std::chrono::steady_clock::now();
                    if (graph_mode) H(hipGraphLaunch(exec,stream));
                    else N(ncclAllReduce(send,recv,n,ncclBfloat16,ncclSum,comm,stream));
                    H(hipStreamSynchronize(stream));
                    elapsed_ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                    H(hipMemcpy(host.data(),recv,n*2,hipMemcpyDeviceToHost));
                    for (size_t i=0;i<n;++i) if (host[i]!=expected_sum(i,seed))
                        throw std::runtime_error("BF16 allreduce bitwise mismatch");
                }
                std::printf("{\"rank\":%d,\"op\":\"allreduce\",\"elements\":%zu,\"graph\":%s,\"repetitions\":20,\"mean_wall_ms\":%.6f,\"passed\":true}\n",rank,n,graph_mode?"true":"false",elapsed_ms/20);std::fflush(stdout);
                if (exec) H(hipGraphExecDestroy(exec));if (graph) H(hipGraphDestroy(graph));
            }
            H(hipFree(send));H(hipFree(recv));
        }
        size_t n=16384;uint8_t *send=nullptr,*recv=nullptr;H(hipMalloc(&send,n));H(hipMalloc(&recv,n*2));
        std::vector<uint8_t> gathered(n*2);
        hipLaunchKernelGGL(fill_bytes,dim3((n+255)/256),dim3(256),0,stream,send,n,rank,1);
        N(ncclAllGather(send,recv,n,ncclUint8,comm,stream));H(hipStreamSynchronize(stream));
        hipGraph_t graph;hipGraphExec_t exec;
        H(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
        N(ncclAllGather(send,recv,n,ncclUint8,comm,stream));
        H(hipStreamEndCapture(stream,&graph));H(hipGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        size_t nodes=0;H(hipGraphGetNodes(graph,nullptr,&nodes));
                    std::vector<hipGraphNode_t> ns(nodes);H(hipGraphGetNodes(graph,ns.data(),&nodes));
                    int types[20]={};for(auto node:ns){hipGraphNodeType t;H(hipGraphNodeGetType(node,&t));if(int(t)<20)types[int(t)]++;}
                    std::printf("NODE_TYPES rank=%d nodes=%zu",rank,nodes);for(int t=0;t<20;++t)if(types[t])std::printf(" type%d=%d",t,types[t]);std::puts("");
        for (int rep=0;rep<20;++rep) {
            int seed=rep+41;
            hipLaunchKernelGGL(fill_bytes,dim3((n+255)/256),dim3(256),0,stream,send,n,rank,seed);
            H(hipGraphLaunch(exec,stream));H(hipStreamSynchronize(stream));
            H(hipMemcpy(gathered.data(),recv,n*2,hipMemcpyDeviceToHost));
            for (int r=0;r<2;++r) for (size_t i=0;i<n;++i)
                if (gathered[r*n+i]!=uint8_t((i*13+r*37+seed)&255)) throw std::runtime_error("Byte allgather mismatch");
        }
        std::printf("{\"rank\":%d,\"op\":\"allgather_bytes\",\"bytes_per_rank\":%zu,\"graph_nodes\":%zu,\"repetitions\":20,\"passed\":true}\n",rank,n,nodes);std::fflush(stdout);
        H(hipGraphExecDestroy(exec));H(hipGraphDestroy(graph));H(hipFree(send));H(hipFree(recv));
        H(hipStreamDestroy(stream));N(ncclCommDestroy(comm));
        std::printf("{\"rank\":%d,\"all_passed\":true}\n",rank);return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr,"rank %d failed: %s\n",rank,e.what());return 1;
    }
}

# 双 R9700 无 P2P 部署与验证

本方案通过引擎源码兼容无 GPU P2P 的主机，无需修改内核或启动配置。小消息使用注册共享主机内存及原生融合内核，大消息使用 RCCL 精确 BF16 求和。仓库默认的原生 P2P 后端保留，混合模式需显式启用。

## 部署步骤

在双卡推理主机上保留旧模型、启动命令和网关配置，并停止占用显卡的旧服务。随后下载源码与 TP=2 权重：

```sh
git clone --branch codex/rccl-compat https://github.com/xmagicer/saber-engine.git
cd saber-engine
export SABER_CONTAINER=docker
export SABER_PACKS="$HOME/models/saber-packs"
tools/build-image.sh
tools/fetch_packs.sh 27b
```

权重为 Saber 的 NVFP4 转 MXFP4 目标模型和 DFlash2 草稿模型，合计约 26.5 GB。使用现成的 TP=2 包，不能将普通 Hugging Face 检查点目录直接作为 `--pack`。下载完成需核对文件散列；实际迁移已核对全部四个包文件的官方 SHA-256。

编译镜像和运行镜像可以分开。先在编译镜像构建 Rust 程序：

```sh
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD":/work -w /work \
  -e CARGO_TARGET_DIR=/work/.cache/target-container \
  --entrypoint bash localhost/saber-dev:latest \
  -c 'cargo build --locked --release -p saber-cli'
```

使用已验证的 HIP/RCCL 运行镜像启动混合模式：

```sh
export SABER_IMAGE=stilldeadcode/vllm-radiance@sha256:45694209177a55a1ab3ba6702fe6e978b1b66a6e66ae3fc066f8d579f7bc4c25
export RCCL_LIB=/opt/rocm/lib/librccl.so
export GPU_MAX_HW_QUEUES=1
HOST=127.0.0.1 PORT=18080 MODEL=qwen3.8-27b-mxfp4 \
  tools/serve_27b.sh --tp 2 --no-build \
  --enable-prefix-caching on --engine-host-ar
curl -fsS http://127.0.0.1:18080/v1/models
```

`RCCL_LIB` 与 `--engine-host-ar` 同时存在时启用混合模式，保留启动器配置的 RCCL SHM 环境。小于 65 个目标隐藏状态行的消息使用原生共享主机内存通信，其余交给 RCCL。模型上下文为 262144，最多 8 个活动序列；启动器保留显式硬件队列覆盖选项。

先检查裸服务的聊天、工具调用、流式、思考、多轮和长上下文，再让现有鉴权网关指向它。保留原来的模型别名和空闲策略。最后检查取消请求、真实自动卸载、双卡休眠和重新加载；回滚时恢复之前保存的启动命令与网关配置，再加载旧服务。停止独立试验可使用 `tools/serve_27b.sh --tp 2 stop`。

## 实际收益与限制

同一主机、相同日间功耗策略（每卡 300W）、相同保存请求下，完整 57 个请求全部通过：

| 指标 | 原 vLLM-MXFP4 | Saber 混合模式 |
|---|---:|---:|
| 单路代码，中位 tok/s | 185.36 | 201.01 |
| 单路长文，中位 tok/s | 136.47 | 158.51 |
| 8 路总吞吐，tok/s | 500.98 | 546.37 |
| 128K 冷输入首 token，秒 | 60.88 | 61.69 |
| 托管服务加载，秒 | 123.08 | 12.54 |

代码和长文分别提升约 8.4% 和 16.2%，8 路吞吐提升约 9.1%。128K 预填充比旧方案慢约 1.3%；单路代码 300、长文 200 tok/s 的目标尚未达到。测速采用关闭思考、温度零、相同请求和固定输出 token 数，不能用并发总吞吐代替单路成绩。

245762 个输入 token 的检索、鉴权网关、工具、流式、思考、多轮、取消均通过。真实空闲测试中，双卡成功休眠，重载约 14.55 秒；原 7200 秒空闲策略已恢复。现有内核与主机日夜功耗策略均保留。

原 vLLM 使用 Quark MXFP4 检查点，Saber 默认包的量化流程不同，不能把全部差异归因于引擎。匹配 Quark 主干的独立对照包可用[转换工具](../tools/weights/README.md)构建，但其输出头仍为 FP8；该对照在代码和长文之间存在取舍，没有作为统一默认模型。

[DSpine 验证工具](../tools/dspine/README.md)已跑通数值比对和无前视信息的离线提案。当前单 GPU 优化提案约需 10.6 ms，每轮通过约 5.74 个代码 token、3.42 个长文 token。它尚未接入在线 TP=2 服务，不能将这些离线数据算作线上加速收益。

通信协议、RCCL 预热流生命周期修复与更完整的验证说明见[英文文档](HOST-COLLECTIVES.md)。

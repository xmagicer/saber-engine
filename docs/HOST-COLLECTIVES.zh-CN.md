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

DSpine 的在线 TP=2 实验已能通过真实 HTTP 请求生成文本。两张卡对五层草稿模型做张量分片，Saber 使用实际提交的 `[2,17,32,47,62]` 层输入特征，在宿主采样后发送 anchor，并由目标模型验算完整链或 15 节点树。匹配 Quark 主干的单请求试验在下表，输出固定 1024 token，每类取三个原始请求的中位数。

| DSpine 实验配置 | 代码 tok/s | 长文 tok/s |
|---|---:|---:|
| 8 行链，eager 草稿 | 156.29 | 120.54 |
| 16 行链，图重放 | 190.48 | 115.59 |
| 16 行树，图重放 | 201.12 | 133.52 |
| 16 行树，合并投影和 GPU 选树 | 203.91 | 137.57 |

最后一组长文每轮提交约 3.66–4.00 token，耗时约 28.2–28.6 ms。短代码样例中，普通链的换行和 ` with` 两项 logprob 相等，树形计算将 ` with` 排在前面，实测分差 0.125。两种验算形状可以产生不同的贪心文本。线上默认仍采用上表长文更快、支持 8 个活动会话的 DFlash2 混合通信配置。实验的启动命令和单序列限制见 [DSpine 工具说明](../tools/dspine/README.md)。

通信协议、RCCL 预热流生命周期修复与更完整的验证说明见[英文文档](HOST-COLLECTIVES.md)。

## 功率设置与原生 B16 实验

以下测试中，两张 R9700 的功率上限均为 300 W，与驱动报告的 `power1_cap_max` 一致。CPU 的 PL1、PL2 均为 181 W，Turbo 开启。在同一组六个单请求上，自动时钟快于强制 `high`。原生 DFlash2 配置复测为代码 200.87、长文 158.01 tok/s，两卡采样峰值分别为 283 W 和 289 W。

原生 B16 现在支持由同一草稿提案驱动的 10 行或 16 行验算图。每次测试均从固定保存的请求实际生成 1024 token，表中为三个请求的中位数。不同目标主干的输出可能不同。

| 目标主干及配置 | 代码 tok/s | 长文 tok/s |
|---|---:|---:|
| NV，原 B8 权重外推至 B16，15 节点 | 210.23 | 155.26 |
| NV，B16 专训权重，9 节点 | 203.61 | 152.12 |
| Quark，B16 专训权重，9 节点 | 215.46 | 153.54 |
| Quark，B16 专训权重，15 节点 | 214.32 | 150.93 |

16 行实现还支持将 48 层 GDN 状态提交合为一次派发，按 16 行记录单独选核。两张卡的状态对照逐位一致。当前机器这项改变的整机收益较小，Quark 15 节点复测为 209.38 / 152.97 tok/s。单序列 B16 的包转换、复核命令与可选启动参数见[权重工具说明](../tools/weights/README.md)。

固定输出测试中，生成更快的组合使用匹配 Quark 主干、B16 专训权重和 8 行运行块。使用前述运行镜像、RCCL 与共享目录配置，可以在独立端口复现。

```sh
TIER=qwen38-27b-quark-matched-core-fp8head DRAFT=qwen38-dflash2-tandem-b16-fp8head \
  MAX_CTX=262144 MAX_SEQS=8 BATCHES=1,2,3,4,8 WIDTHS=1,8 DRAFT_BLOCK=8 \
  HOST=127.0.0.1 PORT=18081 CNAME=saber-tandem-check MODEL=qwen3.8-27b-mxfp4 \
  tools/serve_27b.sh --tp 2 --no-build --engine-host-ar --enable-prefix-caching on
```

完整 57 个请求中的单路三次中位数为代码 **218.34**、长文 **163.77** tok/s，另一次六请求复测分别为 **222.99** 和 **162.52**。8 路合计 **517.62 tok/s**。245,762-token 检索正确。完整测试的 128K 首 token 为 **67.43 秒**，原生产配置同组长请求的复测为 **63.36 秒**。

训练块和运行块可以分别验证。在相同的 Quark 主干与 8 行运行块下，单独按 8 行训练的检查点得到 **217.06 / 151.22 tok/s**。对本次六个请求，按 16 行训练的检查点在 8 行运行时效果更好。

## 定位单轮目标模型验算开销

对同一份 27B NVFP4 目标包和 DFlash2 草稿，复用 TP=2、8 个活动序列、262144-token 上下文的服务配置。用 ROCm 的 `rocprofv3` 附加到独立测试服务中的单个 TP worker，采集单路请求的 kernel trace。以未挂载 profiler 的六个固定请求评估生成速度，每类三个请求，温度为零，每个请求生成 1024 token。验算图仅覆盖目标模型，完整解码轮还包括草稿、提交和调度。

```sh
python3 tools/bench/target_verify_trace.py rank1_kernel_trace.csv --output attributed.json
```

解析器按 GPU、队列和流对 dispatch 排序，以 embedding、输出头和最终行提交定位完整的 542 节点验算图，再核对投影和树形注意力的调用次数。截断的图会被排除，找不到完整图时退出码为 2。各算子族百分比以图内所有 kernel 的执行时间总和为分母。融合核同时执行矩阵乘、跨卡规约和归一化，耗时覆盖三者。

同一代码请求采到 235 个完整验算图，长文请求采到 306 个。下表按单轮计算，图跨度为中位数，各算子族为算术平均值。

| 指标 | 代码，ms | 长文，ms |
|---|---:|---:|
| 目标验算图跨度 | 17.952 | 18.003 |
| MLP gate/up 投影，64 次 | 5.174 | 5.174 |
| 融合输出投影，127 次合计 | 5.696 | 5.741 |
| 其中 MLP down，63 次 | 3.597 | 3.618 |
| 其中 GDN out，48 次 | 1.561 | 1.579 |
| 其中 attention out，16 次 | 0.538 | 0.544 |
| GDN 输入投影对，48 次 | 2.004 | 2.004 |
| 输出词表头 | 1.029 | 1.030 |

按实际 pack 中的权重、组缩放和行参考值计算，64 层 gate/up 每卡每轮合计 3.032 GB。用上表耗时折算约 586 GB/s。这个数是权重字节数除以 kernel 耗时，缓存命中和其它内存流量需要硬件计数器另行测量。

双卡逐位校验覆盖了融合核的打包写回、合并行信号、分割 K 的计算、两种工作组派发顺序、wave 查表，以及 gate/up 的 LDS 权重布局。随后用每类三个固定请求对照，七个改动均未在代码和长文上稳定胜过前后两次原版。在速度最快的 Quark + Tandem 对照上，原版前后两次为 219.85/161.68 和 223.34/161.32 tok/s，gate/up 新布局为 219.25/161.87 tok/s（代码/长文）。各请求的文本及验算次数均与各自对照相同。

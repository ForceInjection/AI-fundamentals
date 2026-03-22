# 在 Mac 上运行 DeepSeek-R1 模型

## 简介

DeepSeek-R1 是一个强大的开源语言模型，支持本地推理，可用于自然语言处理任务，如对话、文本生成等。本教程将指导你在 Mac 上使用 [Ollama](https://ollama.com) 运行 DeepSeek-R1，并介绍如何通过 [Open-WebUI](https://github.com/open-webui/open-webui) 提供 Web 端访问。

## 1. 安装 Ollama

[Ollama](https://ollama.com) 是一个轻量级的 AI 推理框架，支持本地运行 LLM（大型语言模型）。首先，下载并安装 [Ollama](https://ollama.com/download/Ollama-darwin.zip)。

下载后，解压并按照指引安装。

> 如果你希望通过命令行安装，也可以使用 `brew`（如果可用）：
>
```bash
brew install --cask ollama
```

安装完成后，执行以下命令检查 Ollama 是否安装成功：

```bash
ollama --version
```

如果安装成功，会返回类似 ollama 0.x.x 的版本号。

```bash
ollama version is 0.5.7
```

## 2. 下载 DeepSeek-R1 模型

`Ollama` 提供了便捷的模型管理功能，可以直接下载和运行 `DeepSeek-R1` 模型。执行以下命令拉取模型文件：

```bash
ollama pull deepseek-r1:1.5b
```

该命令会自动下载 `DeepSeek-R1 1.5B` 版本的模型，并存储在本地。

## 3. 本地运行 DeepSeek-R1

下载完成后，可以使用以下命令运行 DeepSeek-R1：

```bash
ollama run deepseek-r1:1.5b
```

执行后，终端会进入交互模式，你可以直接输入文本与模型进行对话。例如：

```bash
>>> 请介绍一下屈原
<think>
好的，用户让我介绍一下屈原。首先，我需要回忆一下他的生平事迹。屈原是战国时期著名的诗人，主要作品《离骚》。

接下来，我要考虑他的一生。他是楚国的著名诗人，可能出生在战国时期的某个地方，然后长期从事诗歌创作。他在楚国灭了游子、齐国的齐桓公之后，逐渐下山做诗人的。

关于他的诗作，特别是《离骚》，我需要详细说明它的内容和影响。用户可能对古典诗词感兴趣，所以这部分很重要。

屈原去世后的情况也很关键。他被楚军击败后，流落街头，写下《离骚》。我应该提到他在哪里去世以及之后的生活状态，比如长期从事诗歌创作直到去世前。

还有，他的文学成就和精神影响方面，用户可能对诗人和他们的精神世界感兴趣，所以这些信息也很重要。

最后，可能需要总结一下屈原的重要性和他在中国文化中的地位，让用户有一个全面的了解。
</think>

屈原是中国古代著名的诗人，以其独特的诗歌风格和豪迈的情怀闻名。他是战国时期的楚国著名诗人，主要作品是《离骚》（Book of Disiples）。以下是关于屈原的一些详细介绍：

### 一、生平事迹
屈原出生于战国时期楚国的楚国城邦。他早年漂泊于楚国边缘的游子和楚国的齐桓公两地之间，受到官场腐败的影响，逐渐下山成为诗人。他长期从事诗歌创作，直到后来的流亡中才放下诗歌，开始创作长篇叙事诗《离骚》。

### 二、主要作品
**《离骚》** 是屈原最为著名的作品之一，字幼安。这首诗以豪放不羁的形象风格著称，充满了对国家和命运的强烈感慨。在《离骚》中，屈原表达了对楚国的怨恨、对政治腐败的不满以及对个人命运的反思。

### 三、精神影响
屈原的一生充满了悲剧色彩，但他也留下了深刻的精神遗产。他的诗歌风格豪放飘逸，充满了对人生无常和社会动荡的无奈感。《离骚》不仅是一首诗，更是一种精神的寄托，代表了屈原个人内心的情感和追求。

### 四、去世与影响
屈原在楚国名门后被楚军击败，流落到边关。他写下了《离骚》，尽管最终未能完成，但他的诗歌风格和思想对后来的诗人产生了深远的影响。他的作品成为中国古典文学的重要瑰宝，也激励了无数人去追求个人的精神自由。

### 五、总结
屈原是中国历史上一个具有重要影响的人物，他的诗歌和精神为中国文化的发展奠定了基础。即使在悲剧的结束时刻，他的作品仍然是推动文学前进的动力。

>>> /?
Available Commands:
  /set            Set session variables
  /show           Show model information
  /load <model>   Load a session or model
  /save <model>   Save your current session
  /clear          Clear session context
  /bye            Exit
  /?, /help       Help for a command
  /? shortcuts    Help for keyboard shortcuts

Use """ to begin a multi-line message.

>>> /bye
```

### 4. 通过 Open-WebUI 运行

如果希望使用 `Web` 界面交互模型，可以安装 `Open-WebUI`。该工具提供了一个用户友好的 `Web` 前端，使得 `DeepSeek-R1` 更加易用。

## 4.1 克隆 Open-WebUI 仓库（可选）

```bash
git clone git@github.com:open-webui/open-webui.git
```

这个步骤可选，主要用于查看 Open-WebUI 代码。如果不需要修改代码，可以直接运行容器。

## 4.2 启动 Open-WebUI 容器

使用 `Docker` 运行 `Open-WebUI`：

```bash
docker run -d \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

说明：

* `-p 3000:8080`：将容器的 8080 端口映射到本机 3000 端口；
* `--add-host=host.docker.internal:host-gateway`：允许容器访问宿主机网络；
* `-v open-webui:/app/backend/data`：挂载数据存储目录；
* `--restart always`：确保容器在重启后自动运行；
* `ghcr.io/open-webui/open-webui:main`：拉取 Open-WebUI 的最新版本镜像。

## 5. 访问 Web 界面

启动容器后，在浏览器中打开：

```plaintext
http://127.0.0.1:3000
```

进入 `Web` 界面后，我们可以配置 `Ollama`，并选择 `deepseek-r1:1.5b` 作为默认模型。完成后，即可在 `WebUI` 中使用 `DeepSeek-R1` 进行文本生成和对话。

## 6. 结束运行

如果不再需要 Open-WebUI，可以执行以下命令停止并删除容器：

```bash
docker stop open-webui
docker rm open-webui
```

如果需要删除存储数据：

```bash
docker volume rm open-webui
```

## 7. 常见问题

### 7.1 下载模型速度慢怎么办？

可以尝试使用 `VPN` 或切换网络环境，或者多次重试 `ollama pull` 命令。

### 7.2 端口 3000 被占用怎么办？

可以修改 docker run 命令的端口映射，例如：

```bash
docker run -d -p 4000:8080 ...
```

然后通过 `http://127.0.0.1:4000` 访问 `WebUI`。

7.3 如何更新 `Open-WebUI`？

如果你已经运行了 Open-WebUI，可以执行以下命令更新：

```bash
docker pull ghcr.io/open-webui/open-webui:main
docker stop open-webui
docker rm open-webui
docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main
```

## 8. 参考链接

* [Ollama 官方网站](https://ollama.com)
* [DeepSeek 开源项目](https://github.com/deepseek-ai)
* [Open-WebUI GitHub](https://github.com/open-webui/open-webui)

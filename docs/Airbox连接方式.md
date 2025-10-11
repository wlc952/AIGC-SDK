## 0. 刷机

如果是原厂发来的裸机，需要刷机；若不是裸机，可跳过此步骤。

**刷机步骤：**

1. 准备一张 micro SD 卡和一个读卡器，将 SD 卡格式化为 EXT4 或 FAT32 格式。
2. 下载 [刷机包](https://pan.baidu.com/s/1fKt-qlhDI8hHV0DUPcETFQ?pwd=wg17)，下载完成后解压缩，将所有文件（如 BOOT 等）直接复制到 SD 卡根目录下。
3. 将 SD 卡插入盒子的 micro SD 卡槽，连接电源线，开机后等待约 10 分钟。时间到后，断电并拔出 SD 卡。

**白盒子风扇调速说明：**

如果使用的是白盒子，风扇不转时可用以下命令调节风扇转速。建议将命令添加到 `/etc/rc.local` 文件中：

```bash
sudo busybox devmem 0x50029000 32 0x500 # 0x500 可替换为 0x100~0x900，数值越小风扇转速越高
sudo busybox devmem 0x50029004 32 0xfa0
```

---

## 1. 连接 Airbox

以下说明以黑盒子为例，白盒子的 WAN 口和 LAN 口功能相反，其他操作相同。

**黑盒子（宽度 400px）：**  
<img src="../assets/airbox_b.jpg" alt="黑盒子" width="400"/>

**白盒子（宽度 400px）：**  
<img src="../assets/airbox_w.png" alt="白盒子" width="400"/>

### 1.1 串口连接

Airbox 有一个 type-C 的 debug 口，连接至电脑 USB 口。使用 MobaXterm 等软件，选择串口连接，比特率设置为 115200，用户名和密码均为 `linaro`。如下图所示：

<img src="../assets/airbox_1_1.png" alt="串口连接" width="800"/>

### 1.2 Airbox 作为主机（host）

将 Airbox 的 WAN 口连接到网线，LAN 口连接到电脑。然后在电脑上配置 IP 地址（以 Windows 系统为例）：

1. 打开“设置”->“网络和 Internet”->“更改适配器选项”。
2. 选择“以太网”->“属性”，手动设置 IP 地址为 `192.168.150.2`，子网掩码为 `255.255.255.0`。
3. 连接成功后，Airbox 的 IP 地址为 `192.168.150.1`。

使用 SSH 工具远程连接 Airbox（如 MobaXterm）：新建主机，IP 或主机名填写 `192.168.150.1`，用户名和密码均为 `linaro`。

### 1.3 Airbox 作为客户端（client）

如果没有额外的路由器连接 Airbox，可以通过电脑共享网络给 Airbox。以 Windows 10 为例：

1. 打开“控制面板”->“网络和 Internet”->“网络和共享中心”->“更改适配器设置”。
2. 右键点击要共享的网络（如 WLAN），选择“属性”-“共享”。
3. 勾选“允许其他网络用户通过此计算机的 Internet 连接来连接”，并选择要共享给的网络。

如下图所示：

<img src="../assets/airbox_1_2.png" alt="网络共享" width="400"/>

此时电脑作为主机，IP 为 `192.168.137.1`，Airbox 作为客户端，IP 为同一子网段（如 `192.168.137.x`）。可先通过串口连接，使用 `ip addr` 命令查看 IP 后，再进行 SSH 连接。建议为 Airbox 设置静态 IP，便于后续连接。

**MacOS 连接 Airbox 教程：**  
[链接](https://github.com/zhengorange/airbox_wiki/blob/master/docs/AirBox%E8%BF%9E%E6%8E%A5%E6%96%B9%E5%BC%8F.md)
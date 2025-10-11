# SG2300X TPU 内存大小修改方法

`memory_edit` 是用于修改 SG2300X 系统内存与 TPU 内存分布的工具，方便开发者根据需求调整整机内存分配。

---

## 1. 安装 memory_edit 工具

> 系统出厂自带，如需重新安装可执行以下命令：

```bash
wget https://github.com/radxa-edge/TPU-Edge-AI/releases/download/v0.1.0/memory_edit_V1.6.1.deb
sudo apt install ./memory_edit_V1.6.1.deb
```

---

## 2. 使用方法

### 2.1 检查当前 TPU 内存分配状态

```bash
memory_edit.sh -p bm1684x_sm7m_v1.0.dtb
```

### 2.2 修改 TPU 内存分配

```bash
memory_edit.sh -c -npu 7168 -vpu 2048 -vpp 3072 bm1684x_sm7m_v1.0.dtb
sudo cp /opt/sophon/memory_edit/emmcboot.itb /boot/emmcboot.itb && sync
sudo reboot
```

---

## 3. 参数说明

- `-p [dts 文件名]`：查看当前设备 TPU 内存分布状态
- `-c [dts 文件名]`：修改 TPU 内存分配
- `-npu [数值]`：设置 NPU 内存大小（单位：MB）
- `-vpu [数值]`：设置 VPU 内存大小（单位：MB）
- `-vpp [数值]`：设置 VPP 内存大小（单位：MB）

---

## 4. 使用示例

- 查看当前 TPU 内存分配状态：

  ```bash
  /opt/sophon/memory_edit/memory_edit.sh -p [dts 文件名]
  ```

- 分配各硬件加速处理器内存（单位：MB）：

  ```bash
  /opt/sophon/memory_edit/memory_edit.sh -c -npu 2048 -vpu 2048 -vpp 2048 [dts 文件名]
  ```

---
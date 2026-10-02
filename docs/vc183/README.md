# MobileScan3D vc183 大版本规划包

**适用基线**：vc182 / `0.14.2-consumer-ux2`  
**统一目标版本**：vc183 / `0.15.x` 系列  
**包类型**：工程内规划文档增量，可直接解压到工程根目录  
**是否覆盖源码**：否

## 版本策略

原本分散规划为 vc183–vc188 的功能，统一收敛为一个 vc183 大版本，并拆分为内部阶段：

- **vc183.1 — Guided Scan**：标准扫描引导
- **vc183.2 — Scan Recovery Loop**：缺口检测与补扫闭环
- **vc183.3 — Model Cleanup**：自动模型清理
- **vc183.4 — Project Library**：模型/项目管理
- **vc183.5 — Adaptive Scan**：自动扫描策略
- **vc183.6 — Measure & Share**：测量、AR 与分享

外部可以继续统一称为 **vc183**，内部开发和灰度测试用 `0.15.1` 到 `0.15.6` 标记阶段。

## 推荐版本映射

| 阶段 | versionName 建议 | 主题 |
|---|---|---|
| vc183.1 | 0.15.1-guided-scan | 标准扫描 |
| vc183.2 | 0.15.2-recovery-loop | 缺口补扫 |
| vc183.3 | 0.15.3-model-cleanup | 模型清理 |
| vc183.4 | 0.15.4-project-library | 项目管理 |
| vc183.5 | 0.15.5-adaptive-scan | 自适应扫描 |
| vc183.6 | 0.15.6-measure-share | 测量与分享 |
| vc183 GA | 0.15.0 / 0.15.x-stable | 消费级完整大版本 |

## 解压位置

直接在项目根目录解压后，会新增：

```text
docs/
└── vc183/
    ├── README.md
    ├── VC183_MASTER_PLAN_ZH.md
    ├── VC183_PRODUCT_SPEC_ZH.md
    ├── VC183_TECH_ARCHITECTURE_ZH.md
    ├── VC183_PHASE_PLAN_ZH.md
    ├── VC183_ACCEPTANCE_GATES_ZH.md
    ├── VC183_RELEASE_STRATEGY_ZH.md
    └── VC183_DELTA_MANIFEST.json
```

本包不修改 vc182 代码，适合作为后续开发基线文档直接进入工程。

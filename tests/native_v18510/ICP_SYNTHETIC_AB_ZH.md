# ICP Synthetic A/B 回归

同一组非对称曲面、同一初始 pose 偏差：

## f9bfbdb 原实现

```text
attempted=1 applied=1 inliers=432
rmse=0.00463343
corrT=0.0119253
corrR=1.0359
outT=0.0268779,-0.0211782,0.00287611
final translation residual=0.0343396 m
```

## vc18510

```text
attempted=1 applied=1
pointToPlaneIterations=3
pointToPointFallbacks=0
inliers=432
rmse=7.50956e-08
corrT=0.0260192
corrR=1.20008
outT≈0,0,0
```

这是 synthetic regression，用于证明求解器行为，不等同于真机精度指标。

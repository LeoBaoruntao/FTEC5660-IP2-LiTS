# 本地材料和提交步骤

本项目尚未上传 GitHub，也没有生成或上传 Zoom 录制视频。

## 你需要完成的个人操作

1. 阅读 `submission/Reproducibility_Report.pdf`，确认你能解释实验设计、结果和局限；如课程要求，在封面补上姓名和学号。
2. 用 PowerPoint 打开 `submission/Presentation.pptx`。英文讲稿在 `submission/Presentation_Script.md`，也放在幻灯片备注中。先练习，再用自己的语气讲解。
3. 使用 Zoom 桌面客户端启动你主持的会议，共享幻灯片窗口，选择录制到计算机。先录 20 秒并回放，确认声音和画面正常，再正式录制。
4. 演示应不超过 15 分钟。讲解目标、方法、对比结果、调试问题和局限，并展示 `reproduction/analysis/trials.csv`。不需要现场重新运行耗时的 API 实验。
5. 停止录制并结束会议，等待 Zoom 完成视频转换。回放检查后，将视频放到课程允许的存储服务中，设置老师可访问的权限。
6. 将项目上传到自己的公开 GitHub 仓库，保留 Git 提交历史。不要仅把 ZIP 上传为仓库中的一个文件；老师需要能够查看和运行代码。
7. 在课程平台提交报告 PDF、自己的 GitHub 仓库链接和录制视频链接。幻灯片不能代替录制视频。
8. 用无痕窗口检查两个链接的访问权限，保存课程平台的提交成功截图。

Zoom 操作参考：[官方本地录制说明](https://support.zoom.com/hc/en/article?id=zm_kb&sysparm_article=KB0076922)。如学校账号禁用了本地录制，请根据账号允许的录制方式操作。

## 哪些文件上传

保留原作者代码、许可证、`README.md`、`README_UPSTREAM.md`、`reproduction/`、`submission/` 和 Git 提交记录。
不要上传 `.env`、`.venv/`、`__pycache__/`、`.presentation-build/` 或 `.pdf-preview/`。
`.env` 是你的密钥；老师可用自己的密钥按 README 运行。

报告中的原始参考成绩与新实验使用不同模型和软件版本，不能声称严格复现了完整论文。
这次实验仅覆盖一个固定题目、每组重复三次，不能声称所有填字任务都得到相同性能。

老师写的是 “13 November, midnight”。以课程页面具体时间为准，建议 11 月 12 日晚前完成提交，避免对 midnight 的理解差异。

## 保留提交记录的 GitHub 上传方法

在自己的 GitHub 账号创建一个空的公开仓库，例如 `FTEC5660-IP2-LiTS`，先不要让 GitHub 自动添加 README 或许可证。
本地项目已经保留作者历史和本次修改提交。使用 PowerShell 进入项目目录，将下面 URL 换成你自己的真实仓库地址后逐行运行：

```powershell
cd 'D:\codex写5660作业\5660个人项目2\02_上传到GitHub的代码\LiTS'
git remote add origin https://github.com/你的用户名/FTEC5660-IP2-LiTS.git
git push -u origin main
```

如弹出 GitHub 登录窗口，登录你自己的账号。整理后的副本已将原作者地址命名为 upstream，无需再次改名；如果提示已有 origin，请先运行 `git remote -v` 检查，不要重复操作或使用强制推送。
推送后打开自己的仓库，确认报告、演示材料和原始结果都存在，且没有 `.env`。
课程平台填仓库首页链接，另填你录制完成的视频链接。不要填写原作者仓库链接作为自己的提交。

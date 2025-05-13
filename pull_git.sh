# 清空本地 Git
rm -rf .git

# 初始化新的 Git 仓库
git init

# 配置用户身份
git config --global user.name "ydchen0806"
git config --global user.email "cyd0806@mail.ustc.edu.cn"

# 添加远程仓库
git remote add origin https://github.com/ydchen0806/ads_generation_system.git

# 直接从远程仓库拉取（这样可以避免冲突）
git fetch origin

# 创建本地main分支并跟踪远程main分支
git checkout -b main origin/main

# 如果你有本地更改想要添加
git add .
git commit -m "Add local changes"

# 如果有冲突，解决后再次提交
# git add .
# git commit -m "Resolve conflicts"

# 推送到远程
git push -u origin main
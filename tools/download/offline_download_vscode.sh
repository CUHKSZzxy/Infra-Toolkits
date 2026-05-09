export COMMIT_ID=bf9252a2fb45be6893dd8870c0bf37e2e1766d61

# 1. download vscode pkgs
wget https://update.code.visualstudio.com/commit:${COMMIT_ID}/server-linux-x64/stable -O vscode-server-linux-x64.tar.gz
wget https://update.code.visualstudio.com/commit:${COMMIT_ID}/cli-alpine-x64/stable -O vscode-cli-alpine-x64.tar.gz

# 2. put it to the correct folder
export BASE_PATH=/home/${USER}/.vscode-server
mkdir -P $BASE_PATH
tar -xzvf ./vscode-server-linux-x64.tar.gz -C ./
mkdir -p $BASE_PATH/cli/servers/Stable-$COMMIT_ID/
mv ./vscode-server-linux-x64/ $BASE_PATH/cli/servers/Stable-$COMMIT_ID/server

tar -xzvf ./vscode-cli-alpine-x64.tar.gz
mv ./code $BASE_PATH/code-$COMMIT_ID

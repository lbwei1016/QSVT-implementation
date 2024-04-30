FROM nvcr.io/nvidia/cuquantum-appliance:23.10

RUN cd ~ && \
    wget https://github.com/neovim/neovim/releases/download/v0.9.4/nvim-linux64.tar.gz && \
    tar xzvf nvim-linux64.tar.gz && \
    export PATH="$PATH:~/nvim-linux64/bin/"

RUN wget https://github.com/ryanoasis/nerd-fonts/releases/download/v3.2.1/0xProto.zip && \
    unzip 0xProto.zip -d ~/.local/share/fonts/

RUN git clone https://github.com/NvChad/starter ~/.config/nvim

CMD ["bash"]
FROM pandoc/latex:latest

WORKDIR /project

RUN tlmgr update --self
RUN tlmgr install collection-fontsrecommended

CMD ["documentation.md", "-o", "documentation.pdf"]
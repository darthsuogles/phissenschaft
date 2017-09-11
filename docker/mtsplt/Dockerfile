FROM kalilinux/kali-linux-docker

RUN apt-get update && apt-get install -y \
    metasploit-framework \
    postgresql

RUN service postgresql start
RUN msfdb init
WORKDIR /mtsplt
COPY entrypoint.sh .

ENTRYPOINT ["/mtsplt/entrypoint.sh"]

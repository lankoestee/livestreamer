# 启动easydarwin
./easydarwin/EasyDarwin & sleep 3
# 启动ffmpeg进行推流
ffmpeg -re -stream_loop -1 -i ./video/football.mp4 -codec copy -acodec copy -rtsp_transport tcp -f rtsp rtsp://localhost:10054/football & sleep 1
ffmpeg -re -stream_loop -1 -i ./video/highway.mp4 -codec copy -acodec copy -rtsp_transport tcp -f rtsp rtsp://localhost:10054/highway
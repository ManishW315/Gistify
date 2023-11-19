import time

import yt_dlp as youtube_dl
from gistify.config import Speech2TextConfig, logger

# Reference (HuggingFace Spaces): https://huggingface.co/spaces/sanchit-gandhi/whisper-large-v2


def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>' " </center>"
    return HTML_str


def download_yt_audio(yt_url, filename):
    info_loader = youtube_dl.YoutubeDL()

    try:
        logger.info("Extracting youtube url info.")
        info = info_loader.extract_info(yt_url, download=False)
    except youtube_dl.utils.DownloadError as e:
        logger.error(e)

    file_length = info["duration_string"]
    file_h_m_s = file_length.split(":")
    file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]

    if len(file_h_m_s) == 1:
        file_h_m_s.insert(0, 0)
    if len(file_h_m_s) == 2:
        file_h_m_s.insert(0, 0)
    file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]

    if file_length_s > Speech2TextConfig.YT_LENGTH_LIMIT_S:
        yt_length_limit_hms = time.strftime("%HH:%MM:%SS", time.gmtime(Speech2TextConfig.YT_LENGTH_LIMIT_S))
        file_length_hms = time.strftime("%HH:%MM:%SS", time.gmtime(file_length_s))
        print(f"Maximum YouTube length is {yt_length_limit_hms}, got {file_length_hms} YouTube video.")

    ydl_opts = {"outtmpl": filename, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
        except youtube_dl.utils.ExtractorError as e:
            logger.error(e)

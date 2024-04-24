from pathlib import Path

from pytube import YouTube
import click


@click.command()
@click.argument('video_url')
@click.argument('video_name')
def prepare_video(video_url, video_name):
    try:
        download_video(video_url, video_name)
    except Exception as e:
        click.echo(f"Error downloading video: {str(e)}")


def download_video(video_url, video_name):
    yt = YouTube(video_url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').get_by_resolution("360p")
    stream.download(output_path="data", filename=video_name + ".mp4")
    click.echo("Video downloaded successfully!")


if __name__ == "__main__":
    prepare_video()

# 音楽から動画を生成するPythonコード

from moviepy.editor import *
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2

# 音楽ファイルを読み込む
audio_file = '/Users/inouemoeki/Desktop/Movie/Night_breeze.mp3'
image_file = '/Users/inouemoeki/Desktop/Movie/image.png'  # アップロードした画像

# 音楽データをロード
y, sr = librosa.load(audio_file, sr=None)

# ビート & スペクトログラム解析
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# 音のエネルギーを取得（スペクトログラム）
S = librosa.feature.melspectrogram(y=y, sr=sr)
S_dB = librosa.power_to_db(S, ref=np.max)

# Ensure beat_times and S_dB have the same length
min_length = min(len(beat_times), S_dB.shape[1])
beat_times = beat_times[:min_length]
S_dB = S_dB[:, :min_length]

# 波形を描画して画像として保存
def create_waveform(y, sr, output_image='waveform.png'):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()

create_waveform(y, sr)

# 音楽と画像を組み合わせた動画の作成
audio = AudioFileClip(audio_file)
image = ImageClip(image_file, duration=audio.duration)

# エフェクト適用
def apply_effects(clip, beat_times, S_dB):
    def effect(get_frame):
        def new_frame(t):
            frame = get_frame(t)
            intensity = np.interp(t, beat_times, S_dB.mean(axis=0))  # 音のエネルギーを取得

            # 画像の拡大縮小（音に合わせて）
            scale = 1 + 0.2 * np.sin(2 * np.pi * intensity)
            resized_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            
            # 画像を回転（ビートごとに少し回転）
            angle = int(20 * np.sin(2 * np.pi * intensity))
            center = (resized_frame.shape[1]//2, resized_frame.shape[0]//2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_frame = cv2.warpAffine(resized_frame, M, (resized_frame.shape[1], resized_frame.shape[0]))
            
            return rotated_frame
        return new_frame
    return clip.fl(lambda gf, t: effect(gf)(t))

# エフェクトを適用
video_with_effects = apply_effects(image.set_audio(audio), beat_times, S_dB)

# 最終的な動画を書き出し
video_with_effects.write_videofile('output_videov2.mp4', fps=24)

"""
【使用方法】
1. `Night_breeze.mp3` と `image.png` をプロジェクトフォルダに用意してください。
2. 上記コードを `Movie effect.py` などの名前で保存。
3. ターミナルで `pip install moviepy librosa matplotlib opencv-python`
4. `python "Movie effect.py"` で実行。
5. `output_video.mp4` が生成されます。（音に合わせて画像が動き、回転・拡大縮小）
"""
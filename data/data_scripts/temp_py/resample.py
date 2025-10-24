#!/usr/bin/env python3
"""
Video resampling script with GPU acceleration and parallel processing.
Resamples videos to 24 fps.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,6,7'
import subprocess
import argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json

def get_video_info(video_path):
    """Get video information using ffprobe."""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,width,height,duration',
        '-of', 'json',
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        if 'streams' in info and len(info['streams']) > 0:
            stream = info['streams'][0]
            fps_str = stream.get('r_frame_rate', '0/1')
            num, den = map(int, fps_str.split('/'))
            current_fps = num / den if den != 0 else 0
            return {
                'fps': current_fps,
                'width': stream.get('width'),
                'height': stream.get('height'),
                'duration': stream.get('duration')
            }
    except Exception as e:
        print(f"Error getting info for {video_path}: {e}")
    return None

def check_ffmpeg_available():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def check_gpu_available():
    """Check if NVIDIA GPU is available for encoding."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def resample_video(args_tuple):
    """Resample a single video to target fps."""
    input_path, output_path, target_fps, use_gpu, overwrite, gpu_id = args_tuple
    
    # Debug: Check input file
    if not os.path.exists(input_path):
        return False, input_path, f"input file not found: {input_path}"
    
    # Check if output already exists
    if os.path.exists(output_path) and not overwrite:
        # Verify the file is valid
        if os.path.getsize(output_path) > 0:
            return True, input_path, "already exists"
        else:
            # Remove invalid file
            os.remove(output_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use temporary file to avoid corrupted output
    temp_output = output_path + '.tmp.mp4'
    
    # Try GPU first, then fallback to CPU if GPU fails
    for attempt, try_gpu in enumerate([use_gpu, False] if use_gpu else [False]):
        # Build ffmpeg command
        cmd = ['ffmpeg', '-y']
        
        # Use GPU encoding if available
        if try_gpu:
            # Specify GPU device for hardware acceleration
            if gpu_id is not None:
                cmd.extend(['-hwaccel', 'cuda', '-hwaccel_device', str(gpu_id)])
            cmd.extend(['-i', input_path, '-hide_banner', '-loglevel', 'error'])
            # Try NVENC (NVIDIA GPU encoding)
            cmd.extend([
                '-c:v', 'h264_nvenc',  # Use NVIDIA GPU encoder
                '-preset', 'p4',  # Medium preset for NVENC
                '-b:v', '5M',  # Bitrate
                '-maxrate', '8M',
                '-bufsize', '10M',
            ])
        else:
            cmd.extend(['-i', input_path, '-hide_banner', '-loglevel', 'error'])
            # CPU encoding with libx264
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
            ])
        
        # Set frame rate and copy audio
        cmd.extend([
            '-r', str(target_fps),
            '-c:a', 'aac',  # Re-encode audio to aac for better compatibility
            '-b:a', '128k',
            '-movflags', '+faststart',  # Optimize for streaming
            '-f', 'mp4',  # Force mp4 format
            temp_output
        ])
        
        try:
            # Run ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout per video
            )
            
            if result.returncode == 0 and os.path.exists(temp_output):
                # Verify output file size
                if os.path.getsize(temp_output) > 1000:  # At least 1KB
                    # Move temp file to final location
                    os.rename(temp_output, output_path)
                    encoder = "GPU" if try_gpu else "CPU"
                    return True, input_path, f"success ({encoder})"
                else:
                    # File too small, probably corrupted
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    if not try_gpu or attempt > 0:
                        return False, input_path, "output file too small"
            else:
                # Check if it's a GPU error and we should retry with CPU
                if try_gpu and attempt == 0:
                    if "nvenc" in result.stderr.lower() or "unsupported device" in result.stderr.lower():
                        # Clean up and retry with CPU
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                        continue
                
                error_msg = result.stderr[-300:] if result.stderr else "unknown error"
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                return False, input_path, error_msg
                
        except subprocess.TimeoutExpired:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            if not try_gpu or attempt > 0:
                return False, input_path, "timeout"
        except Exception as e:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            if not try_gpu or attempt > 0:
                return False, input_path, str(e)
    
    return False, input_path, "all attempts failed"

def find_video_files(directory, extensions=None):
    """Find all video files in directory."""
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    
    video_files = []
    for ext in extensions:
        video_files.extend(Path(directory).rglob(f'*{ext}'))
        video_files.extend(Path(directory).rglob(f'*{ext.upper()}'))
    
    return sorted(set(video_files))

def main():
    parser = argparse.ArgumentParser(description='Resample videos to target fps')
    parser.add_argument(
        '--input_dir',
        type=str,
        default='/mnt/nas01-ak/IndividualDirs/wenxueli/Dataset/clips_short_merged',
        help='Input directory containing videos'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/mnt/nas01-ak/IndividualDirs/wenxueli/Dataset/clips_short_merged_fps24',
        help='Output directory for resampled videos (default: input_dir + _fps24)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=24,
        help='Target frames per second (default: 24)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='Number of parallel workers (default: auto-detect)'
    )
    parser.add_argument(
        '--use_gpu',
        action='store_true',
        default=True,
        help='Use GPU acceleration if available (default: True)'
    )
    parser.add_argument(
        '--no_gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Show what would be done without actually processing'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=0,
        help='Maximum number of files to process (default: 100, set to 0 for all files)'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default="3,4,6,7",
        help='Comma-separated GPU IDs to use (e.g., "0,1,2,3"). Will distribute workload across GPUs.'
    )
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    if not check_ffmpeg_available():
        print("❌ 错误: 未找到 ffmpeg！")
        print("\n请先安装 ffmpeg:")
        print("  Ubuntu/Debian: sudo apt install ffmpeg")
        print("  CentOS/RHEL:   sudo yum install ffmpeg")
        print("  Fedora:        sudo dnf install ffmpeg")
        print("  Conda:         conda install -c conda-forge ffmpeg")
        print("  macOS:         brew install ffmpeg")
        return
    else:
        print("✓ ffmpeg 已找到")
    
    # Handle GPU settings
    use_gpu = args.use_gpu and not args.no_gpu
    gpu_ids = []
    
    if use_gpu:
        gpu_available = check_gpu_available()
        if gpu_available:
            # Parse GPU IDs
            if args.gpu_ids:
                gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]
                print(f"✓ GPU detected, will use GPU IDs: {gpu_ids}")
            else:
                gpu_ids = [0]  # Default to GPU 0
                print("✓ GPU detected, will use GPU 0 (default)")
        else:
            print("✗ GPU not available, will use CPU encoding")
            use_gpu = False
    else:
        print("GPU acceleration disabled")
    
    # Set input and output directories
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(str(input_dir) + f'_fps{args.fps}')
    
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target FPS: {args.fps}")
    
    # Find all video files
    print("\nScanning for video files...")
    video_files = find_video_files(input_dir)
    
    if not video_files:
        print("No video files found!")
        return
    
    print(f"Found {len(video_files)} video files")
    
    # Limit number of files to process
    if args.max_files and args.max_files > 0:
        video_files = video_files[:args.max_files]
        print(f"Limiting to first {len(video_files)} files")
    
    # Show some sample files that will be processed
    print(f"\nSample files to process:")
    for i, vf in enumerate(video_files[:5]):
        print(f"  {i+1}. {vf.name}")
    if len(video_files) > 5:
        print(f"  ... and {len(video_files) - 5} more files")
    
    # Prepare tasks
    tasks = []
    for i, video_path in enumerate(video_files):
        # Preserve directory structure
        rel_path = video_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        # Distribute tasks across GPUs in round-robin fashion
        gpu_id = gpu_ids[i % len(gpu_ids)] if gpu_ids else None
        
        tasks.append((
            str(video_path),
            str(output_path),
            args.fps,
            use_gpu,
            args.overwrite,
            gpu_id
        ))
    
    # Show sample output paths
    print(f"\nSample output paths:")
    for i, (inp, outp, _, _, _, gid) in enumerate(tasks[:3]):
        gpu_str = f" (GPU {gid})" if gid is not None else ""
        print(f"  {Path(inp).name} -> {outp}{gpu_str}")
    if len(tasks) > 3:
        print(f"  ... and {len(tasks) - 3} more files")
    
    if args.dry_run:
        print("\n=== DRY RUN ===")
        for input_path, output_path, _, _, _, gid in tasks[:10]:  # Show first 10
            gpu_str = f" (GPU {gid})" if gid is not None else ""
            print(f"{Path(input_path).name} -> {output_path}{gpu_str}")
        if len(tasks) > 10:
            print(f"... and {len(tasks) - 10} more files")
        print(f"\nTotal: {len(tasks)} files would be processed")
        if gpu_ids:
            print(f"Using GPUs: {gpu_ids}")
        return
    
    # Determine number of workers
    if args.num_workers:
        num_workers = args.num_workers
    else:
        # If using GPU, limit workers to avoid GPU memory issues
        if use_gpu:
            # When using multiple GPUs, allow more workers per GPU
            num_gpus = len(gpu_ids) if gpu_ids else 1
            workers_per_gpu = min(4, max(1, cpu_count() // 4 // num_gpus))
            num_workers = workers_per_gpu * num_gpus
        else:
            num_workers = max(1, cpu_count() // 2)
    
    if use_gpu and gpu_ids:
        print(f"Using {num_workers} parallel workers across {len(gpu_ids)} GPUs")
    else:
        print(f"Using {num_workers} parallel workers")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output directory created/verified: {output_dir}")
    
    # Process videos in parallel
    print(f"\nProcessing {len(tasks)} videos...")
    successful = 0
    failed = 0
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(resample_video, tasks),
            total=len(tasks),
            desc="Resampling"
        ))
    
    # Print results
    print("\n=== Results ===")
    failed_files = []
    for success, input_path, message in results:
        if success:
            successful += 1
            if message != "already exists":
                print(f"✓ {Path(input_path).name}: {message}")
        else:
            failed += 1
            failed_files.append((Path(input_path).name, message))
            print(f"✗ {Path(input_path).name}: {message}")
    
    print(f"\nTotal: {len(results)} videos")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nSome videos failed to process. Check the error messages above.")

if __name__ == '__main__':
    main()


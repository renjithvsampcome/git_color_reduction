import streamlit as st
from PIL import Image
import io
import os
import tempfile
import shutil
from typing import Dict, Tuple, List, Optional, Union

import numpy as np

def reduce_colors(image: Image.Image, max_colors: int = 256) -> Image.Image:
    """Reduces the number of colors in an image using quantization.

    Forces conversion to RGB and back to ensure color reduction. Handles different
    image modes including those with transparency.

    Args:
        image: PIL Image object to be processed.
        max_colors: Maximum number of colors in the output image. Defaults to 256.

    Returns:
        A new PIL Image object with reduced colors.

    Raises:
        ValueError: If the image cannot be processed or color reduction fails.
    """
    if image.mode in ['P', 'RGBA', 'LA']:
        if image.mode in ['RGBA', 'LA']:
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[3])
            else:
                background.paste(image, mask=image.split()[1])
            rgb_image = background
        else:
            rgb_image = image.convert('RGB')
    else:
        rgb_image = image.convert('RGB')

    return rgb_image.quantize(
        colors=max_colors,
        method=2,
        dither=Image.FLOYDSTEINBERG
    )

def optimize_frames(frames: List[Image.Image], max_colors: int = 256) -> List[Image.Image]:
    """Optimizes a list of frames by reducing colors while maintaining consistency.

    Args:
        frames: List of PIL Image objects representing animation frames.
        max_colors: Maximum number of colors in the output frames. Defaults to 256.

    Returns:
        List of optimized PIL Image objects.

    Raises:
        ValueError: If frame optimization fails.
    """
    # optimized_first = reduce_colors(frames[0], max_colors)
    
    optimized_frames = []
    for frame in frames:
        reduced_frame = reduce_colors(frame, max_colors)
        optimized_frames.append(reduced_frame)
    
    return optimized_frames

def verify_color_count(image: Image.Image) -> int:
    """Verifies the actual number of unique colors in an image.

    Args:
        image: PIL Image object to analyze.

    Returns:
        Integer representing the number of unique colors in the image.
    """
    if image.mode == 'P':
        used_colors = len(image.getcolors())
        return used_colors
    else:
        rgb_image = image.convert('RGB')
        colors = rgb_image.getcolors(maxcolors=256**3)
        if colors is None:
            return 256**3
        return len(colors)

def get_gif_info(img: Image.Image) -> Dict[str, Union[int, float, str]]:
    """Extracts detailed information from a GIF file.

    Args:
        img: PIL Image object of the GIF file.

    Returns:
        Dictionary containing GIF information with keys:
            - frame_count: Number of frames
            - width: Image width in pixels
            - height: Image height in pixels
            - frame_duration_ms: Duration of each frame in milliseconds
            - fps: Frames per second
            - mode: Color mode
            - format: Image format
    """
    frames = 0
    try:
        while True:
            frames += 1
            img.seek(frames)
    except EOFError:
        pass
    
    img.seek(0)
    width, height = img.size
    duration = img.info.get('duration', 0)
    fps = 1000 / duration if duration else 0
    
    return {
        'frame_count': frames,
        'width': width,
        'height': height,
        'frame_duration_ms': duration,
        'fps': fps,
        'mode': img.mode,
        'format': img.format
    }

def analyze_image_complexity(frame: Image.Image) -> float:
    """Analyzes image complexity based on color variance and edge detection.

    Args:
        frame: PIL Image object to analyze.

    Returns:
        Float between 0 (simple) and 1 (complex) representing image complexity.
    """
    gray = frame.convert('L')
    pixels = np.array(gray)
    
    variance = np.var(pixels) / 255.0
    gradient_x = np.gradient(pixels, axis=1)
    gradient_y = np.gradient(pixels, axis=0)
    edge_intensity = np.sqrt(gradient_x**2 + gradient_y**2)
    edge_density = np.mean(edge_intensity) / 255.0
    
    complexity = (variance + edge_density) / 2
    return min(1.0, complexity)

def estimate_initial_parameters(
    frames: List[Image.Image], 
    original_size: float,
    target_size: float
) -> Dict[str, Union[int, float]]:
    """Estimates optimal compression parameters based on image analysis.

    Args:
        frames: List of PIL Image objects representing animation frames.
        original_size: Original file size in KB.
        target_size: Desired file size in KB.

    Returns:
        Dictionary containing estimated parameters:
            - initial_quality: Estimated quality level (20-100)
            - complexity: Average image complexity (0-1)
            - color_count: Number of unique colors
            - max_colors: Recommended maximum colors for compression
    """
    first_frame = frames[0]
    colors = first_frame.getcolors(maxcolors=256)
    unique_colors = len(colors) if colors else 256
    
    sample_frames = min(5, len(frames))
    complexities = [analyze_image_complexity(frame) for frame in frames[:sample_frames]]
    avg_complexity = sum(complexities) / len(complexities)
    
    compression_ratio = target_size / original_size
    
    max_colors = 256 if compression_ratio >= 0.5 else (128 if compression_ratio >= 0.3 else 64)
    quality_factor = 0.9 if avg_complexity >= 0.6 else (0.8 if avg_complexity >= 0.3 else 0.7)
    initial_quality = max(20, min(100, int(100 * compression_ratio * quality_factor)))
    
    return {
        'initial_quality': initial_quality,
        'complexity': avg_complexity,
        'color_count': unique_colors,
        'max_colors': max_colors
    }

def binary_search_compression(
    frames: List[Image.Image], 
    target_size: float,
    min_quality: int,
    max_quality: int,
    duration: int,
    temp_dir: str
) -> Tuple[bytes, float, int]:
    """Performs binary search to find optimal compression quality.

    Args:
        frames: List of PIL Image objects representing animation frames.
        target_size: Desired file size in KB.
        min_quality: Minimum acceptable quality level (20-100).
        max_quality: Maximum quality level to try (20-100).
        duration: Frame duration in milliseconds.
        temp_dir: Directory for temporary files.

    Returns:
        Tuple containing:
            - Compressed image data as bytes
            - Achieved file size in KB
            - Final quality level used

    Raises:
        ValueError: If compression fails to achieve valid output.
    """
    best_result: Optional[bytes] = None
    best_size = float('inf')
    best_quality: Optional[int] = None
    min_workable_quality = min_quality
    
    while max_quality - min_quality > 2:
        current_quality = (min_quality + max_quality) // 2
        temp_path = os.path.join(temp_dir, f'temp_q{current_quality}.gif')
        
        try:
            frames[0].save(
                temp_path,
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                quality=current_quality,
                duration=duration,
                loop=0
            )
            
            try:
                with Image.open(temp_path) as test_img:
                    test_img.verify()
                current_size = os.path.getsize(temp_path) / 1024
                
                min_workable_quality = current_quality
                
                if abs(current_size - target_size) < abs(best_size - target_size):
                    with open(temp_path, 'rb') as f:
                        best_result = f.read()
                    best_size = current_size
                    best_quality = current_quality
                
                if current_size > target_size:
                    max_quality = current_quality
                else:
                    min_quality = current_quality
                    
            except Exception:
                min_quality = current_quality + 1
                
        except Exception:
            min_quality = current_quality + 1
            
        if min_quality > max_quality:
            break
    
    if best_result is None and min_workable_quality is not None:
        temp_path = os.path.join(temp_dir, f'temp_final.gif')
        frames[0].save(
            temp_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            quality=min_workable_quality,
            duration=duration,
            loop=0
        )
        with open(temp_path, 'rb') as f:
            best_result = f.read()
        best_size = os.path.getsize(temp_path) / 1024
        best_quality = min_workable_quality
    
    if best_result is None:
        raise ValueError("Could not compress GIF to desired size while maintaining validity")
        
    return best_result, best_size, best_quality

def optimize_gif(
    input_bytes: bytes, 
    target_size_kb: float, 
    color_reduction_level: str = 'auto'
) -> Tuple[bytes, float, float, int]:
    """Optimizes a GIF file through color reduction and compression.

    Args:
        input_bytes: Original GIF file data as bytes.
        target_size_kb: Desired output file size in KB.
        color_reduction_level: Level of color reduction to apply.
            Options: 'auto', 'light', 'moderate', 'aggressive'. Defaults to 'auto'.

    Returns:
        Tuple containing:
            - Optimized GIF data as bytes
            - Achieved file size in KB
            - Compression ratio as percentage
            - Final number of colors

    Raises:
        ValueError: If optimization fails or invalid input is provided.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        input_gif = io.BytesIO(input_bytes)
        img = Image.open(input_gif)
        original_size = len(input_bytes) / 1024
        
        original_colors = verify_color_count(img)
        st.write(f"Original color count: {original_colors}")
        
        if target_size_kb >= original_size:
            st.warning("Target size is larger than original. No compression needed.")
            return input_bytes, original_size, 0, 100
        
        frames = []
        try:
            while True:
                frames.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass
        
        max_colors = {
            'aggressive': 32,
            'moderate': 64,
            'light': 128,
            'auto': 32 if target_size_kb / original_size < 0.3 else (
                64 if target_size_kb / original_size < 0.5 else 128
            )
        }[color_reduction_level]
        
        st.write(f"Target color count: {max_colors}")
        
        optimized_frames = optimize_frames(frames, max_colors)
        reduced_colors = verify_color_count(optimized_frames[0])
        st.write(f"Actual color count after reduction: {reduced_colors}")
        
        temp_path = os.path.join(temp_dir, 'temp_check.gif')
        optimized_frames[0].save(
            temp_path,
            save_all=True,
            append_images=optimized_frames[1:],
            optimize=True,
            duration=img.info.get('duration', 100),
            loop=0,
            colors=max_colors
        )
        
        intermediate_size = os.path.getsize(temp_path) / 1024
        if intermediate_size > original_size:
            st.warning(
                "Color reduction resulted in larger file size. Original file preserved."
            )
            return input_bytes, original_size, 0, original_colors
            
        try:
            optimized_data, achieved_size, final_quality = binary_search_compression(
                optimized_frames,
                target_size_kb,
                min_quality=20,
                max_quality=100,
                duration=img.info.get('duration', 100),
                temp_dir=temp_dir
            )
            
            if achieved_size > original_size:
                st.warning(
                    "Compression resulted in larger file size. Original file preserved."
                )
                return input_bytes, original_size, 0, original_colors
            
            compression_ratio = (1 - (achieved_size / original_size)) * 100
            return optimized_data, achieved_size, compression_ratio, reduced_colors
            
        except Exception as e:
            st.error(f"Compression error: {str(e)}")
            return input_bytes, original_size, 0, original_colors
            
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

def main() -> None:
    """Main function for the Streamlit GIF compression application.
    
    Handles the user interface, file upload, compression settings,
    and displays results and statistics.
    """
    st.title("Enhanced GIF Compressor")
    st.write("This GIF compressor uses advanced color reduction and optimization techniques")
    
    uploaded_file = st.file_uploader("Choose a GIF file", type=['gif'])
    
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read()
            original_size = len(file_content) / 1024
            
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            gif_info = get_gif_info(img)
            
            st.write("### Preview and Compression")
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.write("Original GIF:")
                uploaded_file.seek(0)
                st.image(uploaded_file)
            
            # Color reduction options
            color_reduction = st.select_slider(
                "Color Reduction Level",
                options=['light', 'moderate', 'aggressive', 'auto'],
                value='auto',
                help="Controls how aggressively colors are reduced. Auto adjusts based on target size."
            )
            
            # Target size input
            target_size = st.slider(
                "Target Size (KB)", 
                min_value=int(max(original_size * 0.1, 1)),
                max_value=int(original_size),
                value=int(original_size * 0.4)
            )
            
            if st.button("Compress GIF"):
                with st.spinner("Analyzing and compressing..."):
                    try:
                        optimized_data, achieved_size, compression_ratio, quality = optimize_gif(
                            file_content, target_size, color_reduction
                        )
                        
                        if compression_ratio == 0:
                            st.warning("""
                                The GIF appears to be already optimized or further compression would 
                                increase its size. Original file has been preserved.
                            """)
                        
                        with preview_col2:
                            st.write("Compressed GIF:")
                            st.image(io.BytesIO(optimized_data))
                        
                        st.write("### File Dimensions")
                        dim_col1, dim_col2 = st.columns(2)
                        with dim_col1:
                            st.write("**Original Dimensions:**")
                            st.write(f"Width: {gif_info['width']} pixels")
                            st.write(f"Height: {gif_info['height']} pixels")
                            st.write(f"Resolution: {gif_info['width']} × {gif_info['height']}")
                        
                        with dim_col2:
                            compressed_img = Image.open(io.BytesIO(optimized_data))
                            width, height = compressed_img.size
                            st.write("**Compressed Dimensions:**")
                            st.write(f"Width: {width} pixels")
                            st.write(f"Height: {height} pixels")
                            st.write(f"Resolution: {width} × {height}")
                        
                        st.write("### File Details")
                        comparison_col1, comparison_col2 = st.columns(2)
                        
                        with comparison_col1:
                            st.write("**Original File Details:**")
                            st.write(f"📊 Size: {original_size:.2f} KB")
                            st.write(f"🎞️ Frames: {gif_info['frame_count']}")
                            st.write(f"⏱️ Frame Duration: {gif_info['frame_duration_ms']} ms")
                            st.write(f"🎯 FPS: {gif_info['fps']:.1f}")
                            st.write(f"🎨 Color Mode: {gif_info['mode']}")
                        
                        with comparison_col2:
                            st.write("**Compressed File Details:**")
                            st.write(f"📊 Size: {achieved_size:.2f} KB")
                            st.write(f"🎞️ Frames: {gif_info['frame_count']}")
                            st.write(f"⏱️ Frame Duration: {gif_info['frame_duration_ms']} ms")
                            st.write(f"🎯 FPS: {gif_info['fps']:.1f}")
                            st.write(f"⚙️ Quality Level: {quality}")
                            st.write(f"📈 Compression Ratio: {compression_ratio:.1f}%")
                        
                        st.download_button(
                            label="Download Compressed GIF",
                            data=optimized_data,
                            file_name="compressed.gif",
                            mime="image/gif"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during compression: {str(e)}")
                    
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.write("Please try uploading a different GIF file.")

if __name__ == "__main__":
    main()
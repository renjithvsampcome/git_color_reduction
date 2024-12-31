
import streamlit as st
from PIL import Image
import io
import os
import tempfile
import shutil
from typing import Dict, Tuple, List
import numpy as np

def get_gif_info(img):
    """
    Extract detailed information from a GIF file with accurate color counting
    """
    frames = 0
    try:
        while True:
            frames += 1
            img.seek(frames)
    except EOFError:
        pass
    
    # Reset position
    img.seek(0)
    
    # Get other properties
    width, height = img.size
    duration = img.info.get('duration', 0)
    fps = 1000 / duration if duration else 0
    
    # Better color counting
    unique_colors = set()
    for i in range(frames):
        img.seek(i)
        # Convert frame to RGB to get actual colors
        rgb_frame = img.convert('RGB')
        colors = rgb_frame.getcolors(width * height)  # Get all possible colors
        if colors:
            unique_colors.update(color[1] for color in colors)
    
    return {
        'frame_count': frames,
        'width': width,
        'height': height,
        'frame_duration_ms': duration,
        'fps': fps,
        'mode': img.mode,
        'format': img.format,
        'total_colors': len(unique_colors) + 1,
    }

def analyze_image_complexity(frame: Image.Image) -> float:
    """
    Analyze image complexity based on color variance and edge detection
    Returns value between 0 (simple) and 1 (complex)
    """
    # Convert to grayscale for edge detection
    gray = frame.convert('L')
    pixels = np.array(gray)
    
    # Calculate variance in pixel values
    variance = np.var(pixels) / 255.0
    
    # Simple edge detection using gradient
    gradient_x = np.gradient(pixels, axis=1)
    gradient_y = np.gradient(pixels, axis=0)
    edge_intensity = np.sqrt(gradient_x**2 + gradient_y**2)
    edge_density = np.mean(edge_intensity) / 255.0
    
    # Combine metrics
    complexity = (variance + edge_density) / 2
    return min(1.0, complexity)

def estimate_initial_parameters(frames: List[Image.Image], 
                             original_size: float,
                             target_size: float) -> Dict:
    """
    Estimate initial color reduction parameters based on image analysis
    """
    # Analyze first frame for color information
    first_frame = frames[0]
    colors = first_frame.getcolors(maxcolors=256)
    unique_colors = len(colors) if colors else 256
    
    # Calculate average complexity across sample frames
    sample_frames = min(5, len(frames))
    complexities = [analyze_image_complexity(frame) for frame in frames[:sample_frames]]
    avg_complexity = sum(complexities) / len(complexities)
    
    # Calculate initial number of colors based on complexity and size ratio
    compression_ratio = target_size / original_size
    
    # if avg_complexity < 0.3:
    #     color_factor = 0.5  # Simple images can handle more color reduction
    # elif avg_complexity < 0.6:
    #     color_factor = 0.7
    # else:
    #     color_factor = 0.9  # Complex images need more colors
        
    # initial_colors = max(8, min(256, int(unique_colors * compression_ratio * color_factor)))
    
    return {
        'initial_colors': int(compression_ratio * unique_colors),
        'complexity': avg_complexity,
        'original_colors': unique_colors
    }

def binary_search_color_reduction(frames: List[Image.Image], 
                                target_size: float,
                                min_colors: int,
                                max_colors: int,
                                duration: int,
                                temp_dir: str) -> Tuple[bytes, float, int, List[dict]]:
    """
    Simplified binary search for optimal GIF compression using basic optimization only
    Returns: (optimized_gif_bytes, achieved_size, final_colors, compression_log)
    """
    best_result = None
    best_size = float('inf')
    best_colors = None
    min_workable_colors = min_colors
    compression_log = []
    iteration = 0
    
    while max_colors - min_colors > 2:
        iteration += 1
        current_colors = (min_colors + max_colors) // 2
        
        iteration_log = {
            'iteration': iteration,
            'current_colors': current_colors,
            'color_range': f"{min_colors}-{max_colors}"
        }
        
        temp_path = os.path.join(temp_dir, f'temp_c{current_colors}.gif')
        
        try:
            # Convert frames with current color count
            reduced_frames = []
            for frame in frames:
                if current_colors < 32:
                    reduced_frame = frame.convert('P', palette=Image.ADAPTIVE, 
                                                colors=current_colors, 
                                                dither=Image.FLOYDSTEINBERG)
                else:
                    reduced_frame = frame.convert('P', palette=Image.ADAPTIVE, 
                                                colors=current_colors)
                reduced_frames.append(reduced_frame)
            
            # Apply basic optimization
            reduced_frames[0].save(
                temp_path,
                save_all=True,
                append_images=reduced_frames[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
            
            current_size = os.path.getsize(temp_path) / 1024
            
            iteration_log.update({
                'achieved_size': current_size,
                'target_met': current_size <= target_size,
                'size_diff': abs(current_size - target_size)
            })
            
            min_workable_colors = min(current_colors, min_workable_colors or current_colors)
            
            if current_size <= target_size:
                with open(temp_path, 'rb') as f:
                    best_result = f.read()
                best_size = current_size
                best_colors = current_colors
                
                if abs(current_size - target_size) < target_size * 0.1:
                    iteration_log['result'] = 'target_achieved'
                    compression_log.append(iteration_log)
                    return best_result, best_size, best_colors, compression_log
            
            if current_size > target_size:
                max_colors = current_colors
                iteration_log['result'] = 'size_too_large'
            else:
                min_colors = current_colors
                iteration_log['result'] = 'size_acceptable'
                
        except Exception as e:
            iteration_log['error'] = str(e)
        
        compression_log.append(iteration_log)
        
        if min_colors >= max_colors:
            break
    
    # Final optimization attempt if needed
    if best_result is None and min_workable_colors is not None:
        final_attempt_log = {
            'iteration': 'final',
            'current_colors': min_workable_colors
        }
        
        temp_path = os.path.join(temp_dir, f'temp_final.gif')
        try:
            reduced_frames = []
            for frame in frames:
                reduced_frame = frame.convert('P', palette=Image.ADAPTIVE, 
                                           colors=min_workable_colors,
                                           dither=Image.FLOYDSTEINBERG)
                reduced_frames.append(reduced_frame)
            
            reduced_frames[0].save(
                temp_path,
                save_all=True,
                append_images=reduced_frames[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
            
            current_size = os.path.getsize(temp_path) / 1024
            
            final_attempt_log.update({
                'achieved_size': current_size,
                'target_met': current_size <= target_size,
                'size_diff': abs(current_size - target_size)
            })
            
            with open(temp_path, 'rb') as f:
                best_result = f.read()
            best_size = current_size
            best_colors = min_workable_colors
            
            compression_log.append(final_attempt_log)
            
        except Exception as e:
            final_attempt_log['error'] = str(e)
            compression_log.append(final_attempt_log)
    
    if best_result is None:
        raise ValueError("Could not compress GIF to desired size while maintaining validity")
        
    return best_result, best_size, best_colors, compression_log

def optimize_gif(input_bytes: bytes, target_size_kb: float) -> Tuple[bytes, float, float, int, List[dict]]:
    """
    Optimized GIF compression using color reduction with detailed logging
    Returns: (optimized_data, achieved_size, compression_ratio, final_colors, compression_log)
    """
    temp_dir = tempfile.mkdtemp()
    try:
        input_gif = io.BytesIO(input_bytes)
        img = Image.open(input_gif)
        original_size = len(input_bytes) / 1024
        
        # If target size is larger than original, return original
        if target_size_kb >= original_size:
            return input_bytes, original_size, 0, 256, []
            
        frames = []
        try:
            while True:
                frames.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass
            
        params = estimate_initial_parameters(frames, original_size, target_size_kb)
        
        # If image already has very few colors and target size is close to original,
        # return original
        if (params['original_colors'] < 16 and 
            target_size_kb > original_size * 0.8):
            return input_bytes, original_size, 0, params['original_colors'], []
            
        try:
            optimized_data, achieved_size, final_colors, compression_log = binary_search_color_reduction(
                frames,
                target_size_kb,
                min_colors=8,
                max_colors=params['initial_colors'],
                duration=img.info.get('duration', 100),
                temp_dir=temp_dir
            )
            
            # If compressed size is larger than original, return original
            if achieved_size >= original_size:
                return input_bytes, original_size, 0, params['original_colors'], compression_log
                
            compression_ratio = (1 - (achieved_size / original_size)) * 100
            return optimized_data, achieved_size, compression_ratio, final_colors, compression_log
            
        except ValueError as e:
            # If compression fails, return original
            return input_bytes, original_size, 0, params['original_colors'], []
            
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass

def main():
    st.title("Smart GIF Color Reducer")
    st.write("""
    This GIF compressor reduces file size through intelligent color reduction. 
    It preserves all original frames while finding the optimal color palette.
    """)
    
    uploaded_file = st.file_uploader("Choose a GIF file", type=['gif'])
    
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.read()
            original_size = len(file_content) / 1024
            
            uploaded_file.seek(0)
            img = Image.open(uploaded_file)
            gif_info = get_gif_info(img)
            
            # Preview section
            st.write("### Preview and Color Reduction")
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.write("Original GIF:")
                uploaded_file.seek(0)
                st.image(uploaded_file)
                st.write(f"Original Colors: {gif_info['total_colors']}")
            
            # Target size input
            target_size = st.slider(
                "Target Size (KB)", 
                min_value=int(max(original_size * 0.1, 1)),
                max_value=int(original_size),
                value=int(original_size * 0.4)
            )
            
            if st.button("Reduce Colors"):
                with st.spinner("Analyzing and reducing colors..."):
                    try:
                        optimized_data, achieved_size, compression_ratio, colors, compression_log = optimize_gif(
                            file_content, target_size
                        )
                        
                        if compression_ratio == 0:
                            st.warning("""
                                The GIF appears to be already optimized or further color reduction would 
                                not decrease its size. Original file has been preserved.
                            """)
                        
                        with preview_col2:
                            st.write("Compressed GIF:")
                            st.image(io.BytesIO(optimized_data))
                            st.write(f"Reduced Colors: {colors}")
                        
                        # File details comparison
                        st.write("### File Details")
                        comparison_col1, comparison_col2 = st.columns(2)
                        
                        with comparison_col1:
                            st.write("**Original File Details:**")
                            st.write(f"📊 Size: {original_size:.2f} KB")
                            st.write(f"🎨 Colors: {gif_info['total_colors']}")
                            st.write(f"🎞️ Frames: {gif_info['frame_count']}")
                            st.write(f"⏱️ Frame Duration: {gif_info['frame_duration_ms']} ms")
                            st.write(f"🎯 FPS: {gif_info['fps']:.1f}")
                        
                        with comparison_col2:
                            st.write("**Compressed File Details:**")
                            st.write(f"📊 Size: {achieved_size:.2f} KB")
                            st.write(f"🎨 Colors: {colors}")
                            st.write(f"🎞️ Frames: {gif_info['frame_count']}")
                            st.write(f"⏱️ Frame Duration: {gif_info['frame_duration_ms']} ms")
                            st.write(f"📈 Compression Ratio: {compression_ratio:.1f}%")
                        
                        # Compression Details
                        if compression_log:
                            st.write("### Compression Process Details")
                            st.write(f"Total Iterations: {len(compression_log)}")
                            
                            for iteration in compression_log:
                                iteration_num = iteration.get('iteration', 'Final')
                                colors = iteration.get('current_colors', 'N/A')
                                with st.expander(f"Iteration {iteration_num} - Colors: {colors}"):
                                    if 'color_range' in iteration:
                                        st.write(f"Color Range: {iteration['color_range']}")
                                    
                                    if 'error' in iteration:
                                        st.write(f"❌ Error: {iteration['error']}")
                                    else:
                                        st.write(f"Size: {iteration['achieved_size']:.2f} KB")
                                        st.write(f"Target Met: {'✅' if iteration['target_met'] else '❌'}")
                                        st.write(f"Size Difference: {iteration['size_diff']:.2f} KB")
                                    
                                    if 'result' in iteration:
                                        st.write(f"\nIteration Result: {iteration['result']}")
                        
                        st.download_button(
                            label="Download Color-Reduced GIF",
                            data=optimized_data,
                            file_name="color_reduced.gif",
                            mime="image/gif"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during color reduction: {str(e)}")
                        st.exception(e)
                    
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.write("Please try uploading a different GIF file.")

if __name__ == "__main__":
    main()
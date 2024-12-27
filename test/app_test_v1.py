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
        'total_colors': len(unique_colors)
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
    
    if avg_complexity < 0.3:
        color_factor = 0.5  # Simple images can handle more color reduction
    elif avg_complexity < 0.6:
        color_factor = 0.7
    else:
        color_factor = 0.9  # Complex images need more colors
        
    initial_colors = max(8, min(256, int(unique_colors * compression_ratio * color_factor)))
    
    return {
        'initial_colors': initial_colors,
        'complexity': avg_complexity,
        'original_colors': unique_colors
    }

def binary_search_color_reduction(frames: List[Image.Image], 
                                target_size: float,
                                min_colors: int,
                                max_colors: int,
                                duration: int,
                                temp_dir: str) -> Tuple[bytes, float, int, Dict]:
    """
    Enhanced binary search for optimal GIF compression with detailed loop tracking
    Returns: (optimized_gif_bytes, achieved_size, color_count, optimization_stats)
    """
    best_result = None
    best_size = float('inf')
    best_colors = None
    min_workable_colors = min_colors
    
    # Statistics tracking
    stats = {
        'total_iterations': 0,
        'successful_iterations': 0,
        'optimization_attempts': [],
        'color_attempts': set(),
        'best_strategy': None
    }
    
    optimization_strategies = [
        {'optimize': True, 'disposal': 2, 'name': 'dispose_previous'},
        {'optimize': True, 'disposal': 1, 'name': 'keep_previous'},
        {'optimize': True, 'transparency': 0, 'name': 'transparency'},
        {'optimize': True, 'name': 'basic_optimization'}
    ]
    
    # Early termination threshold
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while max_colors - min_colors > 2:
        current_colors = (min_colors + max_colors) // 2
        stats['color_attempts'].add(current_colors)
        
        iteration_info = {
            'colors': current_colors,
            'strategies_tried': [],
            'best_size': None
        }
        
        for strategy in optimization_strategies:
            stats['total_iterations'] += 1
            temp_path = os.path.join(temp_dir, f'temp_c{current_colors}.gif')
            
            try:
                # Print progress information
                print(f"\nAttempting compression with {current_colors} colors using {strategy['name']}")
                print(f"Current range: {min_colors} - {max_colors} colors")
                
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
                
                # Save with current strategy
                reduced_frames[0].save(
                    temp_path,
                    save_all=True,
                    append_images=reduced_frames[1:],
                    duration=duration,
                    loop=0,
                    **{k: v for k, v in strategy.items() if k != 'name'}
                )
                
                current_size = os.path.getsize(temp_path) / 1024
                print(f"Achieved size: {current_size:.2f}KB (Target: {target_size:.2f}KB)")
                
                iteration_info['strategies_tried'].append({
                    'strategy': strategy['name'],
                    'size': current_size
                })
                
                stats['successful_iterations'] += 1
                consecutive_failures = 0
                
                if current_size < best_size:
                    with open(temp_path, 'rb') as f:
                        best_result = f.read()
                    best_size = current_size
                    best_colors = current_colors
                    stats['best_strategy'] = strategy['name']
                    iteration_info['best_size'] = current_size
                
                # Early success exit
                if abs(current_size - target_size) < target_size * 0.1:
                    stats['optimization_attempts'].append(iteration_info)
                    return best_result, best_size, best_colors, stats
                
            except Exception as e:
                print(f"Failed attempt: {str(e)}")
                consecutive_failures += 1
                
                if consecutive_failures >= max_consecutive_failures:
                    print("Too many consecutive failures, adjusting search range")
                    max_colors = current_colors
                    break
                
                continue
        
        stats['optimization_attempts'].append(iteration_info)
        
        # Binary search adjustment
        if best_size > target_size:
            max_colors = current_colors
        else:
            min_colors = current_colors
        
        if min_colors >= max_colors:
            break
    
    # Final optimization attempt if needed
    if best_result is None and min_workable_colors is not None:
        print("\nPerforming final optimization attempt with minimum colors")
        temp_path = os.path.join(temp_dir, f'temp_final.gif')
        
        for strategy in optimization_strategies:
            stats['total_iterations'] += 1
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
                    **{k: v for k, v in strategy.items() if k != 'name'}
                )
                
                current_size = os.path.getsize(temp_path) / 1024
                print(f"Final attempt size: {current_size:.2f}KB")
                
                if best_result is None or current_size < best_size:
                    with open(temp_path, 'rb') as f:
                        best_result = f.read()
                    best_size = current_size
                    best_colors = min_workable_colors
                    stats['best_strategy'] = strategy['name']
                    
            except Exception as e:
                print(f"Final attempt failed with strategy {strategy['name']}: {str(e)}")
                continue
    
    if best_result is None:
        raise ValueError("Could not compress GIF to desired size while maintaining validity")
        
    return best_result, best_size, best_colors, stats

# Update the optimize_gif function to handle the new statistics
def optimize_gif(input_bytes: bytes, target_size_kb: float) -> Tuple[bytes, float, float, int, Dict]:
    """
    Optimized GIF compression using color reduction with detailed statistics
    """
    temp_dir = tempfile.mkdtemp()
    try:
        input_gif = io.BytesIO(input_bytes)
        img = Image.open(input_gif)
        original_size = len(input_bytes) / 1024
        
        if target_size_kb >= original_size:
            return input_bytes, original_size, 0, 256, {'note': 'No optimization needed'}
            
        frames = []
        try:
            while True:
                frames.append(img.copy())
                img.seek(img.tell() + 1)
        except EOFError:
            pass
            
        params = estimate_initial_parameters(frames, original_size, target_size_kb)
        
        if (params['original_colors'] < 16 and target_size_kb > original_size * 0.8):
            return input_bytes, original_size, 0, params['original_colors'], {'note': 'Already optimized'}
            
        try:
            optimized_data, achieved_size, final_colors, stats = binary_search_color_reduction(
                frames,
                target_size_kb,
                min_colors=8,
                max_colors=params['initial_colors'],
                duration=img.info.get('duration', 100),
                temp_dir=temp_dir
            )
            
            if achieved_size >= original_size:
                return input_bytes, original_size, 0, params['original_colors'], {'note': 'Compression ineffective'}
                
            compression_ratio = (1 - (achieved_size / original_size)) * 100
            return optimized_data, achieved_size, compression_ratio, final_colors, stats
            
        except ValueError as e:
            return input_bytes, original_size, 0, params['original_colors'], {'error': str(e)}
            
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
            
            st.write("### Preview and Color Reduction")
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.write("Original GIF:")
                uploaded_file.seek(0)
                st.image(uploaded_file)
                st.write(f"Original Colors: {gif_info['total_colors']}")
            
            target_size = st.slider(
                "Target Size (KB)", 
                min_value=int(max(original_size * 0.1, 1)),
                max_value=int(original_size),
                value=int(original_size * 0.4)
            )
            
            if st.button("Reduce Colors"):
                with st.spinner("Analyzing and reducing colors..."):
                    try:
                        optimized_data, achieved_size, compression_ratio, colors, stats = optimize_gif(
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
                        
                        # New section for optimization statistics
                        st.write("### Optimization Statistics")
                        st.write(f"Total Iterations: {stats.get('total_iterations', 0)}")
                        st.write(f"Successful Iterations: {stats.get('successful_iterations', 0)}")
                        st.write(f"Best Strategy: {stats.get('best_strategy', 'N/A')}")
                        st.write(f"Color Values Tried: {len(stats.get('color_attempts', []))}")
                        
                        # Detailed optimization attempts
                        if 'optimization_attempts' in stats:
                            st.write("### Optimization Progress")
                            for i, attempt in enumerate(stats['optimization_attempts'], 1):
                                st.write(f"**Attempt {i}:**")
                                st.write(f"Colors: {attempt['colors']}")
                                if attempt.get('best_size'):
                                    st.write(f"Best Size Achieved: {attempt['best_size']:.2f}KB")
                                st.write("Strategies tried:")
                                for strategy in attempt.get('strategies_tried', []):
                                    st.write(f"- {strategy['strategy']}: {strategy['size']:.2f}KB")
                        
                        st.download_button(
                            label="Download Color-Reduced GIF",
                            data=optimized_data,
                            file_name="color_reduced.gif",
                            mime="image/gif"
                        )
                        
                    except Exception as e:
                        st.error(f"Error during color reduction: {str(e)}")
                    
        except Exception as e:
            st.error(f"An error occurred while processing the file: {str(e)}")
            st.write("Please try uploading a different GIF file.")

if __name__ == "__main__":
    main()
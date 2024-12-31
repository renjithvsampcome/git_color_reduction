import streamlit as st
from PIL import Image
import io
import os
import tempfile
import shutil
from typing import Dict, Tuple, List
import numpy as np

def get_gif_info(img):
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
    
    unique_colors = set()
    for i in range(frames):
        img.seek(i)
        rgb_frame = img.convert('RGB')
        colors = rgb_frame.getcolors(width * height)
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
    gray = frame.convert('L')
    pixels = np.array(gray)
    variance = np.var(pixels) / 255.0
    gradient_x = np.gradient(pixels, axis=1)
    gradient_y = np.gradient(pixels, axis=0)
    edge_intensity = np.sqrt(gradient_x**2 + gradient_y**2)
    edge_density = np.mean(edge_intensity) / 255.0
    complexity = (variance + edge_density) / 2
    return min(1.0, complexity)

def estimate_initial_parameters(frames: List[Image.Image], original_size: float, target_size: float) -> Dict:
    first_frame = frames[0]
    colors = first_frame.getcolors(maxcolors=256)
    unique_colors = len(colors) if colors else 256
    sample_frames = min(5, len(frames))
    complexities = [analyze_image_complexity(frame) for frame in frames[:sample_frames]]
    avg_complexity = sum(complexities) / len(complexities)
    compression_ratio = target_size / original_size
    
    if avg_complexity < 0.3:
        color_factor = 0.5
    elif avg_complexity < 0.6:
        color_factor = 0.7
    else:
        color_factor = 0.9
        
    initial_colors = max(8, min(256, int(unique_colors * compression_ratio * color_factor)))
    
    return {
        'initial_colors': initial_colors,
        'complexity': avg_complexity,
        'original_colors': unique_colors
    }

def binary_search_color_reduction(frames: List[Image.Image], target_size: float, min_colors: int, max_colors: int, duration: int, temp_dir: str) -> Tuple[bytes, float, int, Dict]:
    best_result = None
    best_size = float('inf')
    best_colors = None
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
    
    while max_colors - min_colors > 2:
        current_colors = (min_colors + max_colors) // 2
        stats['color_attempts'].add(current_colors)
        
        iteration_info = {
            'colors': current_colors,
            'strategies_tried': [],
            'best_size': None,
            'best_data': None
        }
        
        iteration_best_size = float('inf')
        iteration_best_data = None
        
        for strategy in optimization_strategies:
            stats['total_iterations'] += 1
            temp_path = os.path.join(temp_dir, f'temp_c{current_colors}.gif')
            
            try:
                reduced_frames = []
                for frame in frames:
                    reduced_frame = frame.convert('P', palette=Image.ADAPTIVE, colors=current_colors)
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
                
                with open(temp_path, 'rb') as f:
                    current_data = f.read()
                
                strategy_result = {
                    'strategy': strategy['name'],
                    'size': current_size,
                    'data': current_data
                }
                
                iteration_info['strategies_tried'].append(strategy_result)
                stats['successful_iterations'] += 1
                
                if current_size < iteration_best_size:
                    iteration_best_size = current_size
                    iteration_best_data = current_data
                
                if current_size < best_size:
                    best_result = current_data
                    best_size = current_size
                    best_colors = current_colors
                    stats['best_strategy'] = strategy['name']
                    iteration_info['best_size'] = current_size
                
                if abs(current_size - target_size) < target_size * 0.1:
                    iteration_info['best_data'] = iteration_best_data
                    stats['optimization_attempts'].append(iteration_info)
                    return best_result, best_size, best_colors, stats
                
            except Exception as e:
                continue
        
        iteration_info['best_data'] = iteration_best_data
        stats['optimization_attempts'].append(iteration_info)
        
        if best_size > target_size:
            max_colors = current_colors
        else:
            min_colors = current_colors
        
        if min_colors >= max_colors:
            break
    
    return best_result, best_size, best_colors, stats

def optimize_gif(input_bytes: bytes, target_size_kb: float) -> Tuple[bytes, float, float, int, Dict]:
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
    
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    if 'original_info' not in st.session_state:
        st.session_state.original_info = None
    if 'expander_state' not in st.session_state:
        st.session_state.expander_state = True
    
    uploaded_file = st.file_uploader("Choose a GIF file", type=['gif'])
    
    if uploaded_file is not None:
        file_content = uploaded_file.read()
        original_size = len(file_content) / 1024
        
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        gif_info = get_gif_info(img)
        
        st.session_state.original_info = {
            'size': original_size,
            'gif_info': gif_info,
            'content': file_content
        }
        
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
            st.session_state.expander_state = False
            
            with st.spinner("Analyzing and reducing colors..."):
                try:
                    optimized_data, achieved_size, compression_ratio, colors, stats = optimize_gif(
                        file_content, target_size
                    )
                    
                    st.session_state.optimization_results = {
                        'data': optimized_data,
                        'size': achieved_size,
                        'ratio': compression_ratio,
                        'colors': colors,
                        'stats': stats
                    }
                    
                except Exception as e:
                    st.error(f"Error during color reduction: {str(e)}")
        
        if st.session_state.optimization_results:
            with preview_col2:
                st.write("Compressed GIF:")
                st.image(io.BytesIO(st.session_state.optimization_results['data']))
                st.write(f"Reduced Colors: {st.session_state.optimization_results['colors']}")
            
            st.write("### 📊 File Details")
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.write("**📈 Original File Details:**")
                st.write(f"📊 Size: {st.session_state.original_info['size']:.2f} KB")
                st.write(f"🎨 Colors: {st.session_state.original_info['gif_info']['total_colors']}")
                st.write(f"🎞️ Frames: {st.session_state.original_info['gif_info']['frame_count']}")
                st.write(f"⏱️ Frame Duration: {st.session_state.original_info['gif_info']['frame_duration_ms']} ms")
                st.write(f"🎯 FPS: {st.session_state.original_info['gif_info']['fps']:.1f}")
            
            with comp_col2:
                st.write("**📉 Compressed File Details:**")
                st.write(f"📊 Size: {st.session_state.optimization_results['size']:.2f} KB")
                st.write(f"🎨 Colors: {st.session_state.optimization_results['colors']}")
                st.write(f"🎞️ Frames: {st.session_state.original_info['gif_info']['frame_count']}")
                st.write(f"⏱️ Frame Duration: {st.session_state.original_info['gif_info']['frame_duration_ms']} ms")
                st.write(f"📈 Compression Ratio: {st.session_state.optimization_results['ratio']:.1f}%")
            
            st.download_button(
                label="💾 Download Final Optimized GIF",
                data=st.session_state.optimization_results['data'],
                file_name="final_optimized.gif",
                mime="image/gif",
                key="download_final"
            )
            
            stats = st.session_state.optimization_results['stats']
            if 'optimization_attempts' in stats:
                st.write("### 📋 Optimization History")
                for i, attempt in enumerate(stats['optimization_attempts'], 1):
                    attempt_key = f"attempt_{i}"
                    with st.expander(f"🎨 Attempt {i} - {attempt['colors']} colors", expanded=st.session_state.expander_state):
                        attempt_col1, attempt_col2 = st.columns(2)
                        with attempt_col1:
                            if attempt.get('best_data'):
                                st.image(io.BytesIO(attempt['best_data']), use_column_width=True)
                        with attempt_col2:
                            st.write(f"🎯 Colors: {attempt['colors']}")
                            if attempt.get('best_size'):
                                st.write(f"📊 Best Size: {attempt['best_size']:.2f}KB")
                            
                            if attempt.get('best_data'):
                                st.download_button(
                                    label=f"💾 Download",
                                    data=attempt['best_data'],
                                    file_name=f"attempt_{i}_{attempt['colors']}_colors.gif",
                                    mime="image/gif",
                                    key=f"download_{attempt_key}"
                                )

if __name__ == "__main__":
    main()
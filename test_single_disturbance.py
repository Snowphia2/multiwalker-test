"""
单个配置扰动测试脚本
用于测试单个agent和扰动幅度的完整流程
"""
import os
import json
from datetime import datetime
import sys

# 确保能导入项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# #region agent log
# Hypothesis A, B, D, E: Check Python environment and package availability
with open('/root/2507-multiwalker-harl/.cursor/debug.log', 'a') as f:
    import json as json_lib
    f.write(json_lib.dumps({"sessionId":"debug-session","runId":"pre-import","hypothesisId":"A,B,D,E","location":"test_single_disturbance.py:14","message":"Python env check","data":{"python_version":sys.version,"python_executable":sys.executable,"sys_path":sys.path[:3]},"timestamp":int(__import__('time').time()*1000)}) + '\n')
# #endregion

try:
    # #region agent log
    # Hypothesis A, D: Check if hydra can be imported
    with open('/root/2507-multiwalker-harl/.cursor/debug.log', 'a') as f:
        f.write(json_lib.dumps({"sessionId":"debug-session","runId":"pre-import","hypothesisId":"A,D","location":"test_single_disturbance.py:19","message":"Attempting hydra import","data":{},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    import hydra
    
    # #region agent log
    # Hypothesis A: Hydra imported successfully
    with open('/root/2507-multiwalker-harl/.cursor/debug.log', 'a') as f:
        f.write(json_lib.dumps({"sessionId":"debug-session","runId":"pre-import","hypothesisId":"A","location":"test_single_disturbance.py:26","message":"Hydra import success","data":{"hydra_version":str(hydra.__version__) if hasattr(hydra,'__version__') else 'unknown',"hydra_file":hydra.__file__},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
except ImportError as e:
    # #region agent log
    # Hypothesis A, D: Hydra import failed
    with open('/root/2507-multiwalker-harl/.cursor/debug.log', 'a') as f:
        f.write(json_lib.dumps({"sessionId":"debug-session","runId":"pre-import","hypothesisId":"A,D","location":"test_single_disturbance.py:33","message":"Hydra import FAILED","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    raise

try:
    # #region agent log
    # Hypothesis C: Check if omegaconf can be imported
    with open('/root/2507-multiwalker-harl/.cursor/debug.log', 'a') as f:
        f.write(json_lib.dumps({"sessionId":"debug-session","runId":"pre-import","hypothesisId":"C","location":"test_single_disturbance.py:41","message":"Attempting omegaconf import","data":{},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
    from omegaconf import OmegaConf
    
    # #region agent log
    # Hypothesis C: OmegaConf imported successfully
    with open('/root/2507-multiwalker-harl/.cursor/debug.log', 'a') as f:
        f.write(json_lib.dumps({"sessionId":"debug-session","runId":"pre-import","hypothesisId":"C","location":"test_single_disturbance.py:48","message":"OmegaConf import success","data":{"omegaconf_version":str(OmegaConf.__version__) if hasattr(OmegaConf,'__version__') else 'unknown'},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    
except ImportError as e:
    # #region agent log
    # Hypothesis C: OmegaConf import failed
    with open('/root/2507-multiwalker-harl/.cursor/debug.log', 'a') as f:
        f.write(json_lib.dumps({"sessionId":"debug-session","runId":"pre-import","hypothesisId":"C","location":"test_single_disturbance.py:55","message":"OmegaConf import FAILED","data":{"error":str(e),"error_type":type(e).__name__},"timestamp":int(__import__('time').time()*1000)}) + '\n')
    # #endregion
    raise

from sources.skill.code.eval import (
    eval, 
    filter_recovered_episodes, 
    plot_recovered_cases_only,
    print_recovery_stats
)


def test_single_config(agent_id=0, magnitude=0.3):
    """测试单个配置
    
    Args:
        agent_id: 要测试的agent ID (0, 1, 或 2)
        magnitude: 扰动幅度 (0.1 到 0.5)
    """
    print(f"\n{'='*60}")
    print(f"Testing Single Configuration")
    print(f"Agent: {agent_id}, Magnitude: {magnitude}")
    print(f"{'='*60}\n")
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./results/single_disturbance_test/{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # 初始化 wandb（即使禁用也需要初始化以避免错误）
        import wandb
        wandb.init(mode="disabled", project="test")
        
        # 加载基础配置
        with hydra.initialize(config_path="sources/skill/1.config/task/eval", version_base=None):
            cfg = hydra.compose(config_name="default")
            
            # 禁用 struct 模式以允许添加新字段
            OmegaConf.set_struct(cfg, False)
            OmegaConf.set_struct(cfg.environment.env_tweak, False)
            OmegaConf.set_struct(cfg.model, False)
            
            # 修改配置为扰动测试
            cfg.environment.env_tweak.disturbance_mode = 'adaptive'
            cfg.environment.env_tweak.disturb_target_agent = agent_id
            cfg.environment.env_tweak.disturb_magnitude = magnitude
            cfg.environment.env_tweak.disturb_start_step = 100
            
            # 设置 tweak_types 为空，避免路径重复（eval.py会自动添加n_walkers）
            cfg.environment.env_tweak.tweak_types = []
            
            # 使用训练好的模型
            cfg.model.save_group = 'handpicked/mw'
            
            # 禁用渲染 - 我们只需要角度数据来证明恢复
            cfg.eval_settings.functions.render = False
            
            # 增加episodes数量以获得更多recovered cases
            cfg.eval_settings.general.eval_episodes = 50
            
            print("Configuration:")
            print(f"  Disturbance Mode: adaptive")
            print(f"  Target Agent: {agent_id}")
            print(f"  Magnitude: {magnitude}")
            print(f"  Start Step: 100")
            print(f"  Failure Threshold: 6.5°")
            print(f"  Recovery Threshold: 5.0°")
            print(f"  Consecutive Steps for Failure: 10")
            print()
            
            # 执行评估
            print("Starting evaluation...")
            result = eval(cfg)
            
            if result:
                # 保存结果JSON
                json_path = os.path.join(results_dir, "result.json")
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2)
                print(f"\n✓ Results saved to: {json_path}")
                
                # 打印关键指标
                print(f"\nKey Metrics:")
                print(f"  MTTF: {result.get('disturbance_mttf_avg', 'N/A')}")
                print(f"  Recovery Time: {result.get('disturbance_recovery_time_avg', 'N/A')}")
                print(f"  Max Angle: {result.get('disturbance_max_angle', 'N/A'):.2f}°" 
                      if result.get('disturbance_max_angle') else "  Max Angle: N/A")
                print(f"  Terminate Count: {result.get('terminate_cnt', 'N/A')}")
                print(f"  Total Episodes: {result.get('total_episodes', 'N/A')}")
                
                # 保存所有episodes的完整角度数据（新增功能）
                if 'angle_data_grouped' in result:
                    def save_all_episode_angles(angle_data_grouped, terminate_arr, save_path, disturbance_config):
                        """保存所有episodes的完整角度数据（按顺序）"""
                        all_episodes = []
                        
                        for idx, (angles, term_step) in enumerate(zip(angle_data_grouped, terminate_arr)):
                            episode_data = {
                                'episode_idx': idx,
                                'term_step': term_step,
                                'num_angles': len(angles),
                                'angles': [float(a) for a in angles] if len(angles) > 0 else []
                            }
                            all_episodes.append(episode_data)
                        
                        output_data = {
                            'disturbance_config': disturbance_config,
                            'total_episodes': len(all_episodes),
                            'episodes': all_episodes
                        }
                        
                        with open(save_path, 'w') as f:
                            json.dump(output_data, f, indent=2)
                        
                        # 统计
                        with_data = sum(1 for ep in all_episodes if len(ep['angles']) > 0)
                        print(f"\n✓ Saved all episodes angle data: {save_path}")
                        print(f"  Episodes with data: {with_data}/{len(all_episodes)}")
                    
                    # 保存所有episodes角度数据
                    angles_save_path = os.path.join(results_dir, f"all_angles_agent{agent_id}_mag{magnitude}.json")
                    save_all_episode_angles(
                        result['angle_data_grouped'],
                        result['terminate_arr'],
                        angles_save_path,
                        {
                            'target_agent': agent_id,
                            'magnitude': magnitude,
                            'start_step': 100,
                            'mode': 'adaptive'
                        }
                    )
                
                # 过滤并绘制恢复的episodes（每5个episodes一张图，角度取绝对值）
                if 'angle_data_grouped' in result:
                    recovered, episode_details = filter_recovered_episodes(
                        result['angle_data_grouped'],
                        result['terminate_arr'],
                        max_cycles=2000,
                        package_contact_arr=result.get('package_contact_arr', [])
                    )
                    
                    # 创建单独的过滤日志文件，文件名包含扰动参数
                    filter_log_filename = f"filter_log_agent{agent_id}_mag{magnitude}.json"
                    filter_log_path = os.path.join(results_dir, filter_log_filename)
                    
                    # 准备过滤日志数据
                    filter_log_data = {
                        'disturbance_config': {
                            'target_agent': agent_id,
                            'magnitude': magnitude,
                            'start_step': 100,
                            'mode': 'adaptive',
                            'failure_threshold_deg': 6.5,
                            'recovery_threshold_deg': 5.0
                        },
                        'summary': {
                            'total_episodes': len(episode_details),
                            'recovered_count': len(recovered),
                            'filtered_count': len([e for e in episode_details if e['status'] == 'filtered']),
                            'filter_breakdown': {}
                        },
                        'episode_details': episode_details
                    }
                    
                    # 统计各种过滤原因的数量
                    filter_reasons = {}
                    for ep in episode_details:
                        if ep['status'] == 'filtered':
                            reason = ep['filter_reason']
                            if 'Package touched ground' in reason:
                                key = 'package_touched_ground'
                            elif 'Early termination' in reason:
                                key = 'early_termination'
                            elif 'Never unstable' in reason:
                                key = 'never_unstable'
                            elif 'Failed but never recovered' in reason:
                                key = 'failed_no_recovery'
                            else:
                                key = 'other'
                            filter_reasons[key] = filter_reasons.get(key, 0) + 1
                    
                    filter_log_data['summary']['filter_breakdown'] = filter_reasons
                    
                    # 保存过滤日志文件
                    with open(filter_log_path, 'w') as f:
                        json.dump(filter_log_data, f, indent=2)
                    print(f"\n✓ Filter log saved: {filter_log_path}")
                    
                    # 也保存一个人类可读的文本版本
                    txt_log_filename = f"filter_log_agent{agent_id}_mag{magnitude}.txt"
                    txt_log_path = os.path.join(results_dir, txt_log_filename)
                    with open(txt_log_path, 'w') as f:
                        f.write("="*70 + "\n")
                        f.write(f"DISTURBANCE TEST FILTER LOG\n")
                        f.write("="*70 + "\n\n")
                        f.write(f"Configuration:\n")
                        f.write(f"  Target Agent: {agent_id}\n")
                        f.write(f"  Magnitude: {magnitude}\n")
                        f.write(f"  Start Step: 100\n")
                        f.write(f"  Mode: adaptive\n")
                        f.write(f"  Failure Threshold: 6.5°\n")
                        f.write(f"  Recovery Threshold: 5.0°\n\n")
                        
                        f.write("="*70 + "\n")
                        f.write(f"SUMMARY\n")
                        f.write("="*70 + "\n")
                        f.write(f"Total Episodes: {len(episode_details)}\n")
                        f.write(f"Recovered: {len(recovered)}\n")
                        f.write(f"Filtered: {len([e for e in episode_details if e['status'] == 'filtered'])}\n\n")
                        
                        f.write("Filter Breakdown:\n")
                        for reason, count in filter_reasons.items():
                            f.write(f"  {reason}: {count}\n")
                        f.write("\n")
                        
                        f.write("="*70 + "\n")
                        f.write(f"EPISODE DETAILS\n")
                        f.write("="*70 + "\n\n")
                        
                        for ep in episode_details:
                            f.write(f"[Episode {ep['episode_idx']}]\n")
                            f.write(f"  Term step: {ep['term_step']}\n")
                            if ep['max_angle'] is not None:
                                f.write(f"  Max angle: {ep['max_angle']:.2f}°\n")
                            f.write(f"  Status: {ep['status']}\n")
                            f.write(f"  Reason: {ep['filter_reason']}\n")
                            if ep.get('first_unstable_step') is not None:
                                f.write(f"  First unstable step: {ep['first_unstable_step']}\n")
                            if ep.get('last_unstable_step') is not None:
                                f.write(f"  Last unstable step: {ep['last_unstable_step']}\n")
                            if ep.get('unstable_duration') is not None:
                                f.write(f"  Unstable duration: {ep['unstable_duration']} steps\n")
                            if ep['failure_step'] is not None:
                                f.write(f"  Failure step: {ep['failure_step']}\n")
                            if ep['recovery_step'] is not None:
                                f.write(f"  Recovery step: {ep['recovery_step']}\n")
                                f.write(f"  Recovery time: {ep['recovery_time']} steps\n")
                            f.write("\n")
                    
                    print(f"✓ Human-readable log saved: {txt_log_path}")
                    
                    # 保存完整角度数据（所有包含angles字段的episodes）
                    angles_data_filename = f"full_angles_agent{agent_id}_mag{magnitude}.json"
                    angles_data_path = os.path.join(results_dir, angles_data_filename)
                    
                    full_angle_episodes = []
                    for ep in episode_details:
                        if 'angles' in ep:
                            full_angle_episodes.append({
                                'episode_idx': ep['episode_idx'],
                                'status': ep['status'],
                                'filter_reason': ep['filter_reason'],
                                'angles': ep['angles'],
                                'metadata': {
                                    'max_angle': ep['max_angle'],
                                    'first_unstable_step': ep['first_unstable_step'],
                                    'last_unstable_step': ep['last_unstable_step'],
                                    'recovery_step': ep.get('recovery_step')
                                }
                            })
                    
                    if full_angle_episodes:
                        with open(angles_data_path, 'w') as f:
                            json.dump({
                                'disturbance_config': filter_log_data['disturbance_config'],
                                'total_episodes_with_angles': len(full_angle_episodes),
                                'episodes': full_angle_episodes
                            }, f, indent=2)
                        print(f"✓ Full angle data saved: {angles_data_path} ({len(full_angle_episodes)} episodes)")
                    
                    # 绘制恢复episodes
                    plot_recovered_cases_only(recovered, results_dir, {
                        'target_agent': agent_id,
                        'magnitude': magnitude
                    })
                    
                    # 打印统计
                    print_recovery_stats(recovered)
                
                print(f"\n{'='*60}")
                print(f"Test completed successfully!")
                print(f"Results directory: {results_dir}")
                print(f"{'='*60}\n")
                
                return result
            else:
                print("✗ Evaluation returned no results")
                return None
        
    except Exception as e:
        print(f"\n✗ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # 清理Hydra上下文
        try:
            hydra.core.global_hydra.GlobalHydra.instance().clear()
        except:
            pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test single disturbance configuration')
    parser.add_argument('--agent', type=int, default=0, choices=[0, 1, 2],
                        help='Agent ID to test (0, 1, or 2)')
    parser.add_argument('--magnitude', type=float, default=0.3,
                        help='Disturbance magnitude (e.g., 0.1, 0.2, 0.3, 0.4, 0.5)')
    
    args = parser.parse_args()
    
    test_single_config(agent_id=args.agent, magnitude=args.magnitude)

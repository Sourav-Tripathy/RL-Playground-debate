from debate_env import DebateEnvironment
from agent_runner import lit_agent, sci_agent, judge_agent, get_memory_info, cleanup_memory
import os
import time

LOG_DIR = "debate_logs"

# Start with just one simple topic for testing
topics = [
    "Is Consciousness Fundamentally a Physical Process or Beyond Physical Explanation"
]

print("🚀 Starting debate system...")
print(f"💾 Initial memory: {get_memory_info()}")

# Create environment with fewer turns to reduce memory usage
env = DebateEnvironment(topics, max_turns=5)

try:
    for episode in range(len(topics)):
        print(f"\n🌍 Episode {episode + 1}/{len(topics)}")
        
        # Reset environment - handle both old and new interface
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, info = reset_result  # New interface
        else:
            obs = reset_result     # Old interface
            info = {}  # Initialize info
        
        done = False
        
        print(f"📋 TOPIC: {obs['topic']}")
        print("-" * 60)

        turn_count = 0
        max_turns = env.max_turns  # Align with environment's max_turns
        
        while not done and turn_count < max_turns:
            try:
                print(f"💾 Memory before turn {turn_count + 1}: {get_memory_info()}")
                
                # Generate responses with error handling
                print("🎭 Generating literature response...")
                lit_resp = lit_agent(obs['topic'], info['history'])
                cleanup_memory()
                
                print("🔬 Generating science response...")
                sci_resp = sci_agent(obs['topic'], info['history'])
                cleanup_memory()
                
                print("⚖️ Generating judge response...")
                # Create temporary history for judge
                temp_history = info['history'] + [{
                    'turn': env.current_turn + 1,
                    'literature_response': lit_resp,
                    'science_response': sci_resp
                }]
                judge = judge_agent(obs['topic'], temp_history)
                cleanup_memory()
                
                # Step environment with proper action dictionary
                step_result = env.step({
                    'literature_response': lit_resp,
                    'science_response': sci_resp,
                    'judge_feedback': judge
                })
                
                if len(step_result) == 5:
                    # New interface: obs, reward, done, truncated, info
                    obs, reward, done, truncated, info = step_result
                else:
                    # Old interface
                    obs, reward, done, info = step_result[:4]
                    if len(step_result) > 4:
                        info = step_result[-1]
                
                # Display results
                print(f"\n🎭 Turn {info.get('turn', turn_count + 1)}")
                print(f"📘 Literature: {lit_resp[:200]}..." if len(lit_resp) > 200 else f"📘 Literature: {lit_resp}")
                print(f"🔬 Science   : {sci_resp[:200]}..." if len(sci_resp) > 200 else f"🔬 Science   : {sci_resp}")
                print(f"⚖️ Judge     : {judge['reason'][:200]}..." if len(judge['reason']) > 200 else f"⚖️ Judge     : {judge['reason']}")
                print(f"🏅 Scores    : Lit={judge['lit_score']} | Sci={judge['sci_score']}")
                
                turn_count += 1
                
                # Small delay to help with memory management
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error in turn {turn_count + 1}: {e}")
                print(f"🔍 Error details: {type(e).__name__}")
                
                # Try to get some info for logging
                try:
                    info = {
                        'topic': obs['topic'] if isinstance(obs, dict) else str(obs),
                        'turn': turn_count,
                        'error': str(e),
                        'cumulative_rewards': getattr(env, 'cumulative_scores', {'literature': 0, 'science': 0}),
                        'history': info.get('history', [])
                    }
                except:
                    info = {'error': str(e), 'topic': 'Unknown'}
                
                cleanup_memory()
                break

        # Final results
        try:
            if info and 'winner' in info:
                print(f"\n🏁 Debate Complete — Winner: {info['winner'].upper()}")
                print(f"📊 Final Score — Literature: {info['cumulative_scores']['literature']} | Science: {info['cumulative_scores']['science']}")
            else:
                # Fallback for original environment
                lit_score = env.cumulative_scores.get('literature', 0)
                sci_score = env.cumulative_scores.get('science', 0)
                if lit_score > sci_score:
                    winner = 'literature'
                elif sci_score > lit_score:
                    winner = 'science'
                else:
                    winner = 'tie'
                
                print(f"\n🏁 Debate Complete — Winner: {winner.upper()}")
                print(f"📊 Final Score — Literature: {lit_score} | Science: {sci_score}")
                
                # Update info for logging
                info.update({
                    'winner': winner,
                    'cumulative_rewards': {'literature': lit_score, 'science': sci_score},
                    'history': info.get('history', [])
                })
            
            # Save logs
            try:
                if hasattr(env, 'export_logs') and info:
                    if 'cumulative_scores' not in info:
                        info['cumulative_scores'] = {'literature': 0, 'science': 0}
                    txt_path, json_path = env.export_logs(info, LOG_DIR, episode+1)
                    print(f"📜 Logs saved to: {txt_path} and {json_path}")
                else:
                    # Manual log saving
                    os.makedirs(LOG_DIR, exist_ok=True)
                    import json
                    from datetime import datetime
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    log_file = os.path.join(LOG_DIR, f"debate_manual_{episode+1}_{timestamp}.json")
                    
                    with open(log_file, 'w') as f:
                        json.dump(info, f, indent=2, default=str)
                    print(f"📜 Manual log saved to: {log_file}")
                    
            except Exception as log_error:
                print(f"❌ Could not save logs: {log_error}")
        
        except Exception as final_error:
            print(f"❌ Error in final processing: {final_error}")
        
        print(f"💾 Final memory: {get_memory_info()}")
        cleanup_memory()

except KeyboardInterrupt:
    print("\n⏹️ Interrupted by user")
except Exception as e:
    print(f"❌ Fatal error: {e}")
    import traceback
    traceback.print_exc()
finally:
    cleanup_memory()
    print("🧹 Cleanup complete")

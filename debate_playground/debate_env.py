import gymnasium as gym
from gymnasium import spaces
import json
import numpy as np
from datetime import datetime
import os
from typing import Dict, List, Tuple, Any, Optional

class DebateEnvironment(gym.Env):
    """
    Enhanced debate environment with proper state management and reward system
    """
    
    def __init__(self, topics: List[str], max_turns: int = 6, scoring_method: str = "comparative"):
        super(DebateEnvironment, self).__init__()
        
        # Core configuration
        self.topics = topics
        self.max_turns = max_turns
        self.scoring_method = scoring_method  # "comparative" or "absolute"
        
        # Episode state
        self.current_topic_index = 0
        self.current_turn = 0
        self.episode_history = []
        self.cumulative_scores = {'literature': 0, 'science': 0}
        self.turn_scores = []
        
        # Current episode data
        self.current_topic = None
        self.episode_complete = False
        
        # Gymnasium spaces
        self.observation_space = spaces.Dict({
            'topic': spaces.Text(max_length=500),
            'turn': spaces.Discrete(max_turns + 1),
            'history': spaces.Text(max_length=10000),
            'scores': spaces.Box(low=0, high=100, shape=(2,), dtype=np.float32)
        })
        
        self.action_space = spaces.Dict({
            'literature_response': spaces.Text(max_length=1000),
            'science_response': spaces.Text(max_length=1000),
            'judge_feedback': spaces.Dict({
                'lit_score': spaces.Discrete(10, start=1),
                ' sci_score': spaces.Discrete(10, start=1),
                'reason': spaces.Text(max_length=1000)
            })
        })
        
        # Metrics tracking
        self.episode_metrics = {
            'total_episodes': 0,
            'literature_wins': 0,
            'science_wins': 0,
            'ties': 0,
            'average_scores': {'literature': [], 'science': []},
            'average_turn_length': []
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_turn = 0
        self.episode_history = []
        self.cumulative_scores = {'literature': 0, 'science': 0}
        self.turn_scores = []
        self.episode_complete = False
        
        # Get next topic
        if self.current_topic_index >= len(self.topics):
            self.current_topic_index = 0  # Cycle through topics
        
        self.current_topic = self.topics[self.current_topic_index]
        self.current_topic_index += 1
        
        # Create initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Dict) -> Tuple[Dict, np.ndarray, bool, bool, Dict]:
        """
        Execute one step of the debate
        
        Args:
            action: Dict containing 'literature_response', 'science_response', 'judge_feedback'
        """
        if self.episode_complete:
            raise RuntimeError("Episode is complete. Call reset() to start new episode.")
        
        # Validate action
        if not self._validate_action(action):
            return self._get_observation(), np.array([0.0, 0.0]), True, False, {"error": "Invalid action"}
        
        # Extract responses
        lit_response = action.get('literature_response', '')
        sci_response = action.get('science_response', '')
        judge_feedback = action.get('judge_feedback', {})
        
        # Validate judge feedback
        if not self._validate_judge_feedback(judge_feedback):
            judge_feedback = {'lit_score': 5, 'sci_score': 5, 'reason': 'Invalid judge feedback'}
        
        # Create turn record
        turn_record = {
            'turn': self.current_turn + 1,
            'topic': self.current_topic,
            'literature_response': lit_response,
            'science_response': sci_response,
            'judge_feedback': judge_feedback,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store turn
        self.episode_history.append(turn_record)
        self.turn_scores.append({
            'literature': judge_feedback['lit_score'],
            'science': judge_feedback['sci_score']
        })
        
        # Update cumulative scores
        self.cumulative_scores['literature'] += judge_feedback['lit_score']
        self.cumulative_scores['science'] += judge_feedback['sci_score']
        
        # Increment turn
        self.current_turn += 1
        
        # Check if episode is done
        done = self.current_turn >= self.max_turns
        truncated = False
        
        if done:
            self.episode_complete = True
            self._update_episode_metrics()
        
        # Calculate rewards
        rewards = self._calculate_rewards()
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, rewards, done, truncated, info
    
    def _validate_action(self, action: Dict) -> bool:
        """Validate action format"""
        required_keys = ['literature_response', 'science_response', 'judge_feedback']
        return all(key in action for key in required_keys)
    
    def _validate_judge_feedback(self, judge_feedback: Dict) -> bool:
        """Validate judge feedback format"""
        if not isinstance(judge_feedback, dict):
            return False
        
        required_keys = ['lit_score', 'sci_score', 'reason']
        if not all(key in judge_feedback for key in required_keys):
            return False
        
        # Validate score ranges
        try:
            lit_score = int(judge_feedback['lit_score'])
            sci_score = int(judge_feedback['sci_score'])
            return 1 <= lit_score <= 10 and 1 <= sci_score <= 10
        except (ValueError, TypeError):
            return False
    
    def _calculate_rewards(self) -> np.ndarray:
        """Calculate rewards based on scoring method"""
        if not self.turn_scores:
            return np.array([0.0, 0.0])
        
        if self.scoring_method == "comparative":
            # Reward based on relative performance this turn
            last_scores = self.turn_scores[-1]
            lit_score = last_scores['literature']
            sci_score = last_scores['science']
            
            if lit_score > sci_score:
                return np.array([1.0, -0.5])
            elif sci_score > lit_score:
                return np.array([-0.5, 1.0])
            else:
                return np.array([0.0, 0.0])
        
        elif self.scoring_method == "absolute":
            # Reward based on absolute scores
            last_scores = self.turn_scores[-1]
            return np.array([
                last_scores['literature'] / 10.0,
                last_scores['science'] / 10.0
            ])
        
        return np.array([0.0, 0.0])
    
    def _get_observation(self) -> Dict:
        """Get current observation"""
        # Create history string
        history_str = ""
        for turn in self.episode_history:
            history_str += f"Turn {turn['turn']}: "
            history_str += f"Lit: {turn['literature_response'][:100]}... "
            history_str += f"Sci: {turn['science_response'][:100]}... "
            history_str += f"Judge: {turn['judge_feedback']['reason'][:100]}...\n"
        
        return {
            'topic': self.current_topic or "",
            'turn': self.current_turn,
            'history': history_str,
            'scores': np.array([
                self.cumulative_scores['literature'],
                self.cumulative_scores['science']
            ], dtype=np.float32)
        }
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        info = {
            'topic': self.current_topic,
            'current_turn': self.current_turn,
            'max_turns': self.max_turns,
            'cumulative_scores': self.cumulative_scores.copy(),
            'turn_scores': self.turn_scores.copy(),
            'episode_complete': self.episode_complete,
            'history': self.episode_history.copy()
        }
        
        # Add winner information if episode is complete
        if self.episode_complete:
            lit_total = self.cumulative_scores['literature']
            sci_total = self.cumulative_scores['science']
            
            if lit_total > sci_total:
                info['winner'] = 'literature'
                info['margin'] = lit_total - sci_total
            elif sci_total > lit_total:
                info['winner'] = 'science'
                info['margin'] = sci_total - lit_total
            else:
                info['winner'] = 'tie'
                info['margin'] = 0
            
            # Add episode statistics
            info['episode_stats'] = {
                'total_turns': len(self.episode_history),
                'average_lit_score': np.mean([t['literature'] for t in self.turn_scores]) if self.turn_scores else 0,
                'average_sci_score': np.mean([t['science'] for t in self.turn_scores]) if self.turn_scores else 0,
                'score_variance': {
                    'literature': np.var([t['literature'] for t in self.turn_scores]) if self.turn_scores else 0,
                    'science': np.var([t['science'] for t in self.turn_scores]) if self.turn_scores else 0
                }
            }
        
        return info
    
    def _update_episode_metrics(self):
        """Update overall metrics after episode completion"""
        self.episode_metrics['total_episodes'] += 1
        
        lit_total = self.cumulative_scores['literature']
        sci_total = self.cumulative_scores['science']
        
        if lit_total > sci_total:
            self.episode_metrics['literature_wins'] += 1
        elif sci_total > lit_total:
            self.episode_metrics['science_wins'] += 1
        else:
            self.episode_metrics['ties'] += 1
        
        # Update averages
        avg_lit = np.mean([t['literature'] for t in self.turn_scores]) if self.turn_scores else 0
        avg_sci = np.mean([t['science'] for t in self.turn_scores]) if self.turn_scores else 0
        
        self.episode_metrics['average_scores']['literature'].append(avg_lit)
        self.episode_metrics['average_scores']['science'].append(avg_sci)
        self.episode_metrics['average_turn_length'].append(len(self.episode_history))
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics"""
        if self.episode_metrics['total_episodes'] == 0:
            return self.episode_metrics
        
        metrics = self.episode_metrics.copy()
        
        # Calculate win rates
        total = metrics['total_episodes']
        metrics['win_rates'] = {
            'literature': metrics['literature_wins'] / total,
            'science': metrics['science_wins'] / total,
            'ties': metrics['ties'] / total
        }
        
        # Calculate overall averages
        if metrics['average_scores']['literature']:
            metrics['overall_averages'] = {
                'literature': np.mean(metrics['average_scores']['literature']),
                'science': np.mean(metrics['average_scores']['science']),
                'turn_length': np.mean(metrics['average_turn_length'])
            }
        
        return metrics
    
    def export_logs(self, info: Dict, log_dir: str, episode_id: int) -> Tuple[str, str]:
        """Export detailed logs with enhanced format"""
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_filename = f"debate_{episode_id:03d}_{timestamp}"
        
        # Enhanced TXT log
        txt_path = os.path.join(log_dir, base_filename + ".txt")
        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write("=" * 80 + "\n")
            txt_file.write(f"DEBATE LOG - Episode {episode_id}\n")
            txt_file.write("=" * 80 + "\n")
            txt_file.write(f"Topic: {info.get('topic', 'Unknown')}\n")
            txt_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            txt_file.write(f"Total Turns: {len(info.get('history', []))}\n")
            txt_file.write("-" * 80 + "\n\n")
            
            for turn in info.get('history', []):
                txt_file.write(f"TURN {turn['turn']}\n")
                txt_file.write("-" * 40 + "\n")
                txt_file.write(f"📖 Literature Response:\n{turn['literature_response']}\n\n")
                txt_file.write(f"🔬 Science Response:\n{turn['science_response']}\n\n")
                txt_file.write(f"⚖️  Judge Evaluation:\n{turn['judge_feedback']['reason']}\n")
                txt_file.write(f"📊 Scores: Literature={turn['judge_feedback']['lit_score']}/10, Science={turn['judge_feedback']['sci_score']}/10\n")
                txt_file.write("\n" + "=" * 80 + "\n\n")
            
            txt_file.write("FINAL RESULTS\n")
            txt_file.write("-" * 40 + "\n")
            txt_file.write(f"Winner: {info.get('winner', 'N/A').upper()}\n")
            txt_file.write(f"Final Scores: Literature={info.get('cumulative_scores', {}).get('literature', 0)}, Science={info.get('cumulative_scores', {}).get('science', 0)}\n")
            
            if 'episode_stats' in info:
                stats = info['episode_stats']
                txt_file.write(f"Average Scores: Literature={stats['average_lit_score']:.2f}, Science={stats['average_sci_score']:.2f}\n")
                txt_file.write(f"Score Variance: Literature={stats['score_variance']['literature']:.2f}, Science={stats['score_variance']['science']:.2f}\n")
        
        # Enhanced JSON log
        json_path = os.path.join(log_dir, base_filename + ".json")
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(info, json_file, indent=2, ensure_ascii=False)
        
        return txt_path, json_path
    
    def render(self, mode: str = "human"):
        """Render current state"""
        if mode == "human":
            print(f"Topic: {self.current_topic}")
            print(f"Turn: {self.current_turn}/{self.max_turns}")
            print(f"Scores: Literature={self.cumulative_scores['literature']}, Science={self.cumulative_scores['science']}")
            if self.episode_complete:
                info = self._get_info()
                print(f"Winner: {info.get('winner', 'N/A')}")
        
        return None
    
    def close(self):
        """Clean up resources"""
        pass



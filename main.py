import grid2op

def test_environment():
    print("Initializing Grid2Op environment...")
    
    # טעינת סביבת טסט קטנה ומוכרת (רשת חשמל אמיתית ומוקטנת עם 14 תחנות)
    env = grid2op.make("rte_case14_realistic")
    
    print("\n--- Environment Setup Successful ---")
    print(f"Number of substations (Nodes): {env.n_sub}")
    print(f"Number of power lines (Edges): {env.n_line}")
    print(f"Number of power plants: {env.n_gen}")
    
    # איפוס הסביבה וקבלת המצב (State) הראשון
    obs = env.reset()
    
    print("\n--- Testing a Random Action ---")
    # הגרלת פעולה אקראית לחלוטין (כמו Action Noise כאוטי)
    random_action = env.action_space.sample()
    
    # ביצוע הצעד בסימולטור
    obs, reward, done, info = env.step(random_action)
    
    print(f"Reward received: {reward:.2f}")
    
    # בדיקה האם הפעולה האקראית שרפה את הרשת
    if done:
        print("Result: The grid CRASHED on the first random step! (This is expected with random actions)")
    else:
        print("Result: The grid survived the random step. Safe for now.")

if __name__ == "__main__":
    test_environment()
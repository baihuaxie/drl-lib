"""
    Wrappers for retro games
"""



def wrap_deepmind_retro(env, scale=True, frame_stack=4):
    """
    Configure environment for retro games in DeepMind style
    """
    return env
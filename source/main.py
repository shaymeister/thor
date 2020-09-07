import Config
import Vision

def main():
    """starting node for Thor"""

    if args.vision:
        start_vision(args)


# Determine if vision_main.py is being executed
# directly or from another script
if __name__ == "__main__":
    
    main(Config.create_argparser())

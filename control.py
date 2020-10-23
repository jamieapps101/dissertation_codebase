#! /usr/bin/python3

import argparse
import os

def list_to_nice_string(input):
    output=""
    for term in input:
        if len(output)>0:
            output += ","
        output+= term
    return output

def setup_container(options):
    # used to run the setup function, starting the container and downloading any needed data
    print("Not implemented")
    pass

def build_container(options):
    command = ["docker","build"]
    generic_dockerfile_name = "docker_config/Dockerfile_{}".format(options["mode"])
    specific_dockerfile_name = "docker_config/Dockerfile_{}_{}".format(options["container"],options["mode"]) 
    if os.path.isfile(specific_dockerfile_name):
        command.append("-f {}".format(specific_dockerfile_name))
    else:
        command.append("-f {}".format(generic_dockerfile_name))
        
    command.append("-t {}:{}".format(options["tag"],options["version"]))
    command.append("./docker_config/build_context")
    command_str = " ".join(command)
    # print("command_str:\n{}".format(command_str))
    if options["dry_run"] is not None:
        print("command_str:\n{}".format(command_str))
    else:
        os.system(command_str)  

def run_container(options):
    command = ["docker","run"]
    command.append("--mount type=bind,src={},target=/app".format(   os.path.join(os.getcwd(), "persistent_storage/{}/app".format(options["container"])).replace(" ","\\ ")     ))
    if options["mode"] == "gpu":
        command.append("--gpus all")
    if options["display"] is not None or options["interactive"] is not None:
        command.append("-it")
    command.append("--rm")
    command.append("--env-file docker_config/env_file.txt")
    command.append("{}:{}".format(options["tag"],options["version"]))
    if options["script"] == "bash":
        command.append("bash")
    else:
        if options["interactive"] is not None:
            command.append("python3 -i /app/src/app.py")
        else:
            command.append("python3 /app/src/app.py")
    command_str = " ".join(command)
    # print("command_str:\n{}".format(command_str))
    if options["dry_run"] is not None:
        print("command_str:\n{}".format(command_str))
    else:
        os.system(command_str)   



def main():
    parser = argparse.ArgumentParser(description='Convinience script for controlling docker instances')
    parser.add_argument('command', metavar='C', type=str, nargs=1,
                    help='See read me for command structure')
    parser.add_argument('--tag', metavar='T', type=str, nargs=1,
                    help='specify custom tag', default="jamie/nlp_environment_gpu")
    parser.add_argument('--version', metavar='V', type=str, nargs=1,
                    help='specify custom version', default="1.0")
    parser.add_argument('--script', metavar='S', type=str, nargs=1,
                    help='specify script to run in container, or "bash" for shell', default="app.py")
    parser.add_argument('--interactive', metavar='I', action='store_const', const=1,
                    help='Create interactive session in container')
    parser.add_argument('--display', action='store_const', const=1,
                    help='Print output of container to be run to screen')
    parser.add_argument('--dry_run', action='store_const', const=1,
                    help='Print docker command but don\'t run it')

    args = vars(parser.parse_args())

    print(args)
    # exit()
    # basic command arg check
    command = args["command"][0]
    parts = command.split(".")
    if len(parts) != 3:
        print("command arg incorrect length, see README")
        return

    possible_options = {
        "function":  ["build","run","setup"],
        "mode":      ["cpu","gpu"],
        "container": ["speech2text","text2topic","topic2command"]
    }
    # check args
    for i,term in enumerate(list(possible_options)):
        if not parts[i] in possible_options[term]:
            print("Part of command arg incorrect, see README")
            print("{} should be one of {}".format(parts[i],list_to_nice_string(possible_options[term])))
            return

    options = {
        "mode":parts[1],
        "container":parts[2],
        "tag": args["tag"],
        "version": args["version"],
        "script": args["script"][0],
        "interactive": args["interactive"],
        "dry_run": args["dry_run"],
        "display": args["display"],
    }

    if parts[0] == "build":
        build_container(options)

    if parts[0] == "run":
        run_container(options)

    if parts[0] == "setup":
        setup_container(options)
    

if __name__=="__main__":
    main()    


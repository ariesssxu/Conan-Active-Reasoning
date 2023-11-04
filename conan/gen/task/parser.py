import conan.playground.constants as constants
import conan.gen.task.task_constants as task_constants
import numpy as np
import random

# sub_task begins with "[" or end with "]" are flags for question generating


def parse_get_task(task, inventory_buff=None):
    task = task if task != "wood" else "tree"
    task = task if task != "beef" else "cow"
    task = task if task != "apple" else "apple_tree"
    needed = constants.collect[task]["require"]
    task = "water" if task == "drink" else task
    if not needed:
        if task not in ["cow", "skeleton", "zombie"]:
            return ["[get_" + task, "goto_" + task, "do_" + task, "]"]
        else:
            return parse_defeat_task(task, inventory_buff)
    else:
        sub_tasks = ["[get_" + task]
        for item, num in needed.items():
            if inventory_buff[item] > 0:
                continue
            sub_tasks += parse_make_task(item, inventory_buff)
        sub_tasks.append("goto_" + task)
        sub_tasks.append("do_" + task)
        sub_tasks.append("]")
        return sub_tasks


def parse_make_task(task, inventory_buff=None):
    needed = constants.make[task]["uses"]
    nearby = constants.make[task]["nearby"]
    if not needed and not nearby:
        inventory_buff[task] += 1
        return ["[make_" + task, "make_" + task, "]"]
    else:
        sub_tasks = ["[make_" + task]
        for item, num in needed.items():
            for i in range(num):
                sub_tasks += parse_get_task(item, inventory_buff)
        for item, num in needed.items():
            for item in nearby:
                sub_tasks += parse_place_task(item, inventory_buff)
        sub_tasks.append("make_" + task)
        sub_tasks.append("]")
        inventory_buff[task] += 1
        return sub_tasks


def parse_place_task(task, inventory_buff=None):
    needed = constants.place[task]["uses"]
    if not needed:
        return ["[place_" + task, "place_" + task, "]"]
    else:
        sub_tasks = ["[place_" + task]
        for item, num in needed.items():
            if item in ["bed"]:
                for i in range(num):
                    sub_tasks += parse_make_task(item, inventory_buff)
            else:
                for i in range(num):
                    sub_tasks += parse_get_task(item, inventory_buff)
        sub_tasks.append("place_" + task)
        sub_tasks.append("]")
        return sub_tasks


def parse_defeat_task(task, inventory_buff=None):
    weapon = random.choice([None] + task_constants.weapon)
    if not weapon:
        return ["[defeat_" + task, "chase_" + task, "]"]
    else:
        sub_tasks = ["[defeat_" + task]
        sub_tasks += parse_make_task(weapon, inventory_buff)
        sub_tasks.append("chase_" + task)
        sub_tasks.append("]")
        return sub_tasks


def parse_task(task_type, task):
    inventory_buff = {
        name: info['initial'] for name, info in constants.items.items()}
    sub_tasks = globals()["parse_" + task_type](task, inventory_buff)
    return sub_tasks


# test code
if __name__ == "__main__":
    print(parse_task("make_task", "steak"))

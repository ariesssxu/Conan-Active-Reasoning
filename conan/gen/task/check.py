import numpy as np
import conan.playground.constants as constants


def task_check(task_type, task, env, number=1):

    task_type = task_type.split("_")[0]

    if task_type == "get" or task_type == "make":
        if task in ["water", "lava"]:
            return env.player.inventory[task + "_bucket"] >= number
        return env.player.inventory[task] >= number

    elif task_type == "defeat":
        mat_ids = {x: i for i, x in enumerate([None] + constants.materials)}
        target_id = mat_ids[task + "_dead"]
        mat_map = env.get_detailed_view()[0]
        targets = np.where(mat_map == target_id)
        if len(targets[0]) == 0:
            return False
        else:
            targets = np.array(list(zip(targets[0], targets[1])))
            # find the nearest one with self._pos
            idx = np.abs((targets - env._player.pos)
                         ).sum(axis=1).argmin(axis=0)
            if np.abs((targets[idx] - env._player.pos)).sum() <= 3:
                return True

    elif task_type == "place":
        mat_ids = {x: i for i, x in enumerate([None] + constants.materials)}
        target_id = mat_ids[task]
        mat_map = env.get_detailed_view()[0]
        targets = np.where(mat_map == target_id)
        if len(targets[0]) == 0:
            return False
        else:
            targets = np.array(list(zip(targets[0], targets[1])))
            # find the nearest one with self._pos
            idx = np.abs((targets - env._player.pos)
                         ).sum(axis=1).argmin(axis=0)
            if np.abs((targets[idx] - env._player.pos)).sum() <= 3:
                return True
    return False


def cando_check(mat_map, obj_map, pos):
    if mat_map[pos] in [1, 3, 6, 7] \
            or obj_map[pos] in [2, 4, 6]:
        return True
    return False


def valid_check(pos):
    if pos[0] < 0 or pos[0] >= 64 or pos[1] < 0 or pos[1] >= 64:
        return False
    return True

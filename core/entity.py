"""实体索引与消歧。

单用户场景下取消了「按人分区」（逻辑层 §6）：第三方人、宠物、地点、作品
统一作为**场景里的实体**。理由不是妥协，是更贴近人——人的记忆本来就是围绕
事物、人物、地点组织的语义网络，不是按「这话是谁对我说的」分区。

**记谁不记谁（2026-09-25）**：只记**和用户有个人关系的**——家人 / 朋友 /
同事 / 宠物 / 常住地与老家 / 用户的项目与作品 / 用户喜欢的东西，判据是对话里
说出来的关系或态度（「我妈」「我崇拜他」），每项带一句 `relation`。
公共人物、公共事件、技术名词、话题词都不记（特朗普只有在「我崇拜他」这种
语境里才值得一笔）——所以库里不会再有「设计架构」「写代码」这类标签。
抽取层（`prompts._SCENE_PROMPT` + `scene._norm_entities`）与写入层
（`link_entities`）各有一道闸，判据同一条。

这一层干两件事：
  - **写入侧**：`link_entities` 把场景里的实体挂进索引（消歧）。
  - **读取侧**：`match_known_entities` + `recall_by_entities` 构成**实体旁路**——
    消息里出现「小明」时，最可靠的线索就是这两个字，不必也不该绕道语义检索。
    唤醒层 §5 把它列为旁路而非主路径，正是这个意思。

消歧的保守原则：**只做精确匹配 + 别名，不做自动语义归并。**
「两个小明是不是同一个人」是需要判断的问题，猜错的代价（两条线索被悄悄并成一条）
比新建的代价（索引里多一个条目，**人工可合**）大得多。

那个"人工可合"的口子：实体页每行的「并到…」按钮 →
`weave.merge_entities_confirmed` → `store.merge_entities`（留痕、不可撤销）。
合并会把旧名字**并成别名**——不并的话，以后提到他原来的叫法就命中不了，
等于白合。
"""
# ---------------------------------------------------------------------
# 模块速查
#   层级    ：L4 写入侧（实体）
#   上游    ：model（Scene）
#   下游    ：distill（写入时挂索引）、recall（读取时的实体旁路）、salvage（重记时重新挂）
#   对外入口：`link_entities`、`match_known_entities`、`recall_by_entities`
#   边界    ：不管"什么名字才算同一个"的语义判断（精确匹配在 `store.find_entity`）
# ---------------------------------------------------------------------
from __future__ import annotations

from .model import Scene


def link_entities(scene_id: str, entities: list[dict], store) -> list[str]:
    """把场景里的实体挂进索引，返回命中的 entity_id 列表。

    先精确匹配 name / aliases，匹配不到才新建。别名（aliases）不会自动学习——
    「我家猫」和后来的「它」指同一个，需要消歧判断，第一版不做；
    留 `add_entity(aliases=...)` 与实体页的「并到…」按钮给人工补
    （后者就是 `weave.merge_entities_confirmed`）。

    **门槛（2026-09-25）**：没有 `relation` 的项不建不挂——"这个人和用户有
    什么关系"答不出来，就不是用户的世界里的实体（公共人物、技术名词、
    话题词都卡在这）。抽取层的 `scene._norm_entities` 已经先滤过一遍，
    这里再挡一道：**直接调本函数的路径（打捞重记、未来的导入）同样受约束**。
    """
    ids: list[str] = []
    for item in entities or []:
        name = str((item or {}).get("name") or "").strip()
        if not name:
            continue
        relation = str((item or {}).get("relation") or "").strip()
        if not relation:
            continue
        kind = str((item or {}).get("kind") or "person").strip() or "person"
        found = store.find_entity(name)
        if found is None:
            eid = store.add_entity(name, kind)
        else:
            eid = found.id
        store.link_scene_entity(scene_id, eid, relation)
        ids.append(eid)
    return list(dict.fromkeys(ids))


def match_known_entities(text: str, store, max_hits: int = 5) -> list[str]:
    """从一句话里找出**已知实体**（唤醒层 §5 的实体旁路入口）。

    只匹配库里已有的实体名 / 别名——不匹配未知名字：
    突然出现一个没见过的名字，说明这是新信息，不是「想起什么」的线索。
    按名字长度降序匹配：先长后短，避免「小明」把「小明明」的命中吃掉。
    """
    if not text:
        return []
    hits: list[str] = []
    candidates: list[tuple[str, str]] = []          # (匹配词, 规范名)
    for e in store.all_entities():
        candidates.append((e.name, e.name))
        for alias in e.aliases or []:
            if alias:
                candidates.append((alias, e.name))
    candidates.sort(key=lambda x: len(x[0]), reverse=True)

    lowered = text.lower()
    for word, canonical in candidates:
        if not word:
            continue
        if word.lower() in lowered and canonical not in hits:
            hits.append(canonical)
            if len(hits) >= max_hits:
                break
    return hits


def recall_by_entities(names: list[str], store, limit: int = 10) -> list[Scene]:
    """实体旁路检索：命中的场景直接进候选，**不经语义检索**。

    与 C1 语义检索并行，结果同样进候选池参与四键排序（`recall` 内部的
    `order()`）——两条路的产物是同一类东西，只是发现方式不同。
    """
    return store.scenes_by_entities(names, limit=limit)

"""
群聊触发控制器 (trigger_guard)

解决 AstrBot 内置能力的几个缺口：
1. 群聊主动回复只有一个全局概率(0-1)，写在配置文件里，无法按群/按用户单独配置，
   而且"每条消息独立掷骰子"这种伯努利概率模型对使用者来说很反直觉——0.02 的概率
   听起来很低，但连续几百条消息里完全可能连续触发好几次，也可能很长时间不触发，
   波动大、难预期。
2. 没有"拉黑"（保留上下文但禁止触发 LLM）和"完全屏蔽"（连上下文都不记录）的区分。
3. 唤醒只支持前缀 / @ / 引用，不支持"消息包含某关键词即唤醒"。

因此本插件的主动回复不是"概率"，而是"计数间隔"：给每个作用域（群 / 用户）配置一个
[min, max] 区间（默认 50~100），每次触发后从区间里重新随机一个目标值，攒够这么多条
普通消息后就主动回复一次，然后重新随机下一轮目标——更符合"这个群大概每 50~100 条
消息主动说一次话"的直觉，波动也比纯概率模型小。

实现方式（纯插件，不修改 AstrBot 核心代码）：

- Handler A（guard_message，挂在 AdapterMessageEvent / platform_adapter_type(ALL)，高优先级）：
  在 WakingCheckStage 判定阶段就通过一个恒真 filter 保活流水线（与内置
  group_chat_context 的技巧相同），在 ProcessStage 里对每条群消息做出判断：
    - 命中"完全屏蔽"名单 -> event.should_call_llm(True) + event.stop_event()，
      彻底终止事件传播，本条消息不会被后面任何 handler（包括
      group_chat_context 的上下文记录）看到，天然做到"剔除上下文"。
    - 命中"拉黑"名单 -> 不拦截事件传播（上下文记录等后续 handler 正常执行），
      仅仅不主动去触发 LLM；真正的拦截交给 Handler B 在请求 LLM 前统一兜底。
    - 命中唤醒关键词 -> 设置 is_at_or_wake_command，复用 AstrBot 默认的
      LLM 请求链路（同前缀/@ 唤醒完全一致的后续行为：人设、身份、LTM 等
      都会正常注入）。
    - 否则先检查这个群是不是在"主动回复白名单"里——不在的话直接跳过，不跑触发
      判定（白名单为空则不对任何群主动回复，这是有意为之的默认行为，避免开了
      开关就对机器人在的所有群都开始计数）。
    - 在白名单里的群，按 本群用户区间 > 全局用户区间(跨群) > 群区间 > 全局默认
      区间 的优先级取一个 [min, max]，做一次计数触发判定（见 _should_trigger）。

- Handler B（guard_llm_request，挂在 on_llm_request）：
  这是"请求 LLM 前"的最终闸口。无论 LLM 请求是由默认唤醒链路、
  Handler A 的主动回复、还是 group_chat_context 自身的主动回复触发的，
  只要 sender 在拉黑或屏蔽名单里，这里统一 stop_event()，请求不会真正
  发给 Provider。这一层保证"该用户不能触发 LLM"在任何触发路径下都成立。

强制接管：如果 TriggerGuard 自己的触发判定(active_reply_enable)和 AstrBot 内置的
provider_ltm_settings.active_reply.enable 同时打开，两边会各自独立判定，导致实际
触发频率明显偏离 TriggerGuard 里配置的数值。所以只要 active_reply_enable 为真，
就会强制把内置开关写成 False（见 _force_disable_core_active_reply），并在日志里
说明原因，而不是只提示不处理。

运行时数据（黑名单/屏蔽名单/触发区间覆盖/唤醒关键词）保存在插件数据目录下的
overrides.json，直接编辑该文件即可生效（下一次收到消息时按 mtime 自动重载，
无需重启）。计数器/随机目标值本身是纯内存状态，不落盘，重启后重新计数，这是
预期行为（只是节奏状态，不是配置数据）。

除 wake_keywords 外，其余数据都以协议实例 ID（event.get_platform_id()，同一
协议类型可能同时跑多个机器人实例）为前缀分区，例如 blacklist 条目是
"协议实例ID:QQ号"。旧版本（无协议前缀、且概率是 0-1 浮点数）的数据会在加载时
自动迁移：无协议前缀的归入 "_legacy" 协议桶，浮点概率按 1/p 近似换算成
[min, max] 区间，需要在 UI 设置页里确认/调整。

webchat 是 AstrBot 自带的网页测试对话，没有"群"的概念，也不需要触发控制，
get_platforms() 会主动过滤掉它，不出现在 UI 的协议列表里。
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.platform import MessageType
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, StarTools

_LEGACY_PLATFORM_BUCKET = "_legacy"
_FALLBACK_INTERVAL_MIN = 50
_FALLBACK_INTERVAL_MAX = 100

_DEFAULT_DATA: dict[str, Any] = {
    "_readme": (
        "所有 key/条目都以协议实例 ID 为前缀（同一协议类型可能同时跑多个机器人实例，"
        "用 event.get_platform_id() 区分），可在 UI 设置页 -> 侧边协议列表里查看当前"
        "已启动的协议实例。旧版本(无协议前缀、且是 0-1 浮点概率)的数据会在下次加载时"
        "自动迁移，请在 UI 里确认。"
        "group_interval: {\"协议ID:群号\": {min,max}}(这个群大概每 min~max 条消息主动说一次话); "
        "user_interval: {\"协议ID:群号:QQ号\": {min,max}}(用户在某个群下的区间，优先级最高); "
        "user_interval_global: {\"协议ID:QQ号\": {min,max}}(用户跨所有群的区间，不绑定具体群，"
        "优先级高于 group_interval、低于 user_interval); "
        "blacklist: [\"协议ID:QQ号\",...] 拉黑-保留上下文但不可触发LLM; "
        "blocklist: [\"协议ID:QQ号\",...] 完全屏蔽-不记录上下文且不可触发LLM; "
        "active_reply_whitelist: [\"协议ID:群号\",...] 主动回复白名单-只有在这里的群才会跑"
        "触发区间计数，不在名单里的群哪怕配了 group_interval/user_interval 也不会主动回复"
        "（白名单为空 = 不对任何群主动回复）; "
        "group_max_length: {\"协议ID:群号\": 字数}(这个群单条消息超过这个字数就整条丢弃，"
        "0或不配=不限制); "
        "user_max_length: {\"协议ID:群号:QQ号\": 字数}(用户在某个群下的字数上限，优先级最高); "
        "user_max_length_global: {\"协议ID:QQ号\": 字数}(用户跨所有群的字数上限，优先级高于"
        "group_max_length、低于 user_max_length); "
        "wake_keywords: [关键词,...] 消息中包含即唤醒(不要求前缀/@，全局生效，不分协议)"
    ),
    "group_interval": {},
    "user_interval": {},
    "user_interval_global": {},
    "blacklist": [],
    "blocklist": [],
    "active_reply_whitelist": [],
    "group_max_length": {},
    "user_max_length": {},
    "user_max_length_global": {},
    "wake_keywords": [],
}


def _clamp_interval(lo: Any, hi: Any, default_min: int, default_max: int) -> dict[str, int]:
    try:
        lo = int(lo)
    except (TypeError, ValueError):
        lo = default_min
    try:
        hi = int(hi)
    except (TypeError, ValueError):
        hi = default_max
    lo = max(1, lo)
    hi = max(lo, hi)
    return {"min": lo, "max": hi}


def _probability_to_interval(prob: Any, default_min: int, default_max: int) -> dict[str, int]:
    """把旧版本 0-1 的浮点概率近似换算成 [min, max] 计数区间（均值 ~= 1/p）。"""
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return {"min": default_min, "max": default_max}
    if p <= 0:
        return {"min": default_min, "max": default_max}
    if p >= 1:
        return {"min": 1, "max": 1}
    mean_n = max(1, round(1 / p))
    lo = max(1, round(mean_n * 0.75))
    hi = max(lo + 1, round(mean_n * 1.25))
    return {"min": lo, "max": hi}


def _normalize_interval_value(value: Any, default_min: int, default_max: int) -> dict[str, int]:
    """兼容加载：新格式是 {min,max} 对象，旧格式是 0-1 浮点概率。"""
    if isinstance(value, dict):
        return _clamp_interval(
            value.get("min", default_min), value.get("max", default_max),
            default_min, default_max,
        )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _probability_to_interval(value, default_min, default_max)
    return {"min": default_min, "max": default_max}


def _migrate_flat_interval_data(
    raw: dict[str, Any], default_min: int, default_max: int,
) -> dict[str, dict[str, int]]:
    """用于 group_interval / user_interval_global：key 是 "协议ID:某ID"。
    旧格式 key 没有协议前缀（裸 ID），归入 _legacy 协议桶。"""
    migrated: dict[str, dict[str, int]] = {}
    for key, value in raw.items():
        key = str(key)
        platform_key = key if ":" in key else f"{_LEGACY_PLATFORM_BUCKET}:{key}"
        migrated[platform_key] = _normalize_interval_value(value, default_min, default_max)
    return migrated


def _migrate_user_interval_data(
    raw: dict[str, Any], default_min: int, default_max: int,
) -> dict[str, dict[str, int]]:
    """用于 user_interval：key 是 "协议ID:群号:QQ号"（至少 2 个冒号）。
    旧格式 key 是 "群号:QQ号"（只有 1 个冒号），归入 _legacy 协议桶。"""
    migrated: dict[str, dict[str, int]] = {}
    for key, value in raw.items():
        key = str(key)
        platform_key = key if key.count(":") >= 2 else f"{_LEGACY_PLATFORM_BUCKET}:{key}"
        migrated[platform_key] = _normalize_interval_value(value, default_min, default_max)
    return migrated


def _migrate_entries(entries: set[str]) -> set[str]:
    """旧格式条目是裸 QQ 号（没有协议前缀），归入 _legacy 协议桶。"""
    return {
        entry if ":" in entry else f"{_LEGACY_PLATFORM_BUCKET}:{entry}"
        for entry in entries
    }


def _migrate_flat_int_data(raw: dict[str, Any]) -> dict[str, int]:
    """用于 group_max_length / user_max_length_global：key 是 "协议ID:某ID"，
    值是非负整数字数上限。旧格式 key 没有协议前缀（裸 ID），归入 _legacy 协议桶；
    值非法（非数字/负数）的条目直接丢弃，而不是硬凑一个默认值掩盖配置错误。"""
    migrated: dict[str, int] = {}
    for key, value in raw.items():
        key = str(key)
        try:
            length = max(0, int(value))
        except (TypeError, ValueError):
            continue
        platform_key = key if ":" in key else f"{_LEGACY_PLATFORM_BUCKET}:{key}"
        migrated[platform_key] = length
    return migrated


def _migrate_user_int_data(raw: dict[str, Any]) -> dict[str, int]:
    """用于 user_max_length：key 是 "协议ID:群号:QQ号"（至少 2 个冒号）。
    旧格式 key 是 "群号:QQ号"（只有 1 个冒号），归入 _legacy 协议桶。"""
    migrated: dict[str, int] = {}
    for key, value in raw.items():
        key = str(key)
        try:
            length = max(0, int(value))
        except (TypeError, ValueError):
            continue
        platform_key = key if key.count(":") >= 2 else f"{_LEGACY_PLATFORM_BUCKET}:{key}"
        migrated[platform_key] = length
    return migrated


def _sync_whitelist_with_group_interval(
    group_interval: dict[str, Any], whitelist: set[str],
) -> set[str]:
    """给一个群配了"群聊触发区间"，就默认这个群也该进"主动回复白名单"——不然
    "配了区间但忘了加白名单"会是最常见的误用（实测踩过一次）。只增不减：删掉
    区间覆盖不会把群从白名单里移出去，因为那个群可能仍想用全局默认区间。"""
    return whitelist | set(group_interval.keys())


class TriggerGuard(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context)
        self._config = config or {}

        self._data_path = StarTools.get_data_dir() / "overrides.json"
        self._data: dict[str, Any] = dict(_DEFAULT_DATA)
        self._mtime: float = 0.0
        self._load(initial=True)

        # 触发节奏计数器，纯内存状态，不落盘。key -> 已累计的普通消息数 / 本轮随机目标值。
        self._counters: dict[str, int] = {}
        self._thresholds: dict[str, int] = {}

        self.page_api = None
        self._register_page_api_if_available()

        self._force_disable_core_active_reply()

        if self._enable and self._active_reply_enable:
            wl_count = len(self._data["active_reply_whitelist"])
            if wl_count == 0:
                logger.warning(
                    "[TriggerGuard] 主动回复(计数触发)已启用，但“主动回复白名单”为空——"
                    "当前不会对任何群主动回复。需要在 UI 设置页里把想启用的群加进白名单，"
                    "这是有意为之的默认行为（白名单模式：只有名单里的群才跑）。",
                )
            else:
                logger.info(
                    f"[TriggerGuard] 主动回复(计数触发)已启用，白名单内共 {wl_count} 个群生效，"
                    f"默认区间 {self._global_interval_min}~{self._global_interval_max} 条消息，"
                    "群/用户单独设置的区间以 overrides.json / UI 设置页为准。",
                )
        elif self._enable:
            logger.info(
                "[TriggerGuard] 主动回复(计数触发)当前未启用——群/用户触发区间不会生效，"
                "只有黑名单/屏蔽/关键词唤醒会工作。如需按消息数主动触发，请在 "
                "插件 -> TriggerGuard -> 配置 里打开 active_reply_enable。",
            )

        if self._debug_log:
            logger.info(
                "[TriggerGuard] debug_log 已开启：每条群消息的触发判定过程都会以 INFO "
                "级别打印，排查完建议关掉，避免大群刷屏。",
            )

        logger.info(
            f"[TriggerGuard] 插件已加载，数据文件: {self._data_path}",
        )

    def _register_page_api_if_available(self) -> None:
        """按需注册官方插件页面 API（UI 设置页），旧版 AstrBot 没有该能力时静默跳过。"""
        if not hasattr(self.context, "register_web_api"):
            logger.info(
                "[TriggerGuard] 当前 AstrBot 版本不支持插件 Pages / register_web_api，"
                "UI 设置页不可用，仍可直接编辑 overrides.json。",
            )
            return
        try:
            from .page_api import TriggerGuardPageApi

            self.page_api = TriggerGuardPageApi(self)
            self.page_api.register_routes()
        except Exception as e:
            self.page_api = None
            logger.warning(f"[TriggerGuard] 注册插件页面 API 失败: {e}", exc_info=True)

    # ------------------------------------------------------------------ #
    # 配置 / 数据加载
    # ------------------------------------------------------------------ #

    def _cfg(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    @property
    def _enable(self) -> bool:
        return bool(self._cfg("enable", True))

    @property
    def _active_reply_enable(self) -> bool:
        return bool(self._cfg("active_reply_enable", False))

    @property
    def _global_interval_min(self) -> int:
        try:
            return max(1, int(self._cfg("global_trigger_interval_min", _FALLBACK_INTERVAL_MIN)))
        except (TypeError, ValueError):
            return _FALLBACK_INTERVAL_MIN

    @property
    def _global_interval_max(self) -> int:
        lo = self._global_interval_min
        try:
            hi = int(self._cfg("global_trigger_interval_max", _FALLBACK_INTERVAL_MAX))
        except (TypeError, ValueError):
            hi = _FALLBACK_INTERVAL_MAX
        return max(lo, hi)

    @property
    def _debug_log(self) -> bool:
        return bool(self._cfg("debug_log", False))

    @property
    def _global_max_length(self) -> int:
        try:
            return max(0, int(self._cfg("global_max_length", 0)))
        except (TypeError, ValueError):
            return 0

    def _dlog(self, message: str) -> None:
        """路由每条消息级别的诊断日志：`debug_log` 打开时用 INFO（不用改 AstrBot
        全局日志级别就能看到排查过程），关闭时降级为 DEBUG（默认安静，不刷屏）。
        真正触发主动回复、命中黑/白名单这类低频事件不走这个函数，始终是 INFO。"""
        if self._debug_log:
            logger.info(message)
        else:
            logger.debug(message)

    def _force_disable_core_active_reply(self) -> None:
        """当本插件启用了自己的触发判定时，强制关闭 AstrBot 内置的
        provider_ltm_settings.active_reply.enable，避免两边各自独立判定导致实际
        触发频率明显偏离 TriggerGuard 里设置的数值。"""
        if not self._enable or not self._active_reply_enable:
            return
        try:
            cfg = self.context.get_config()
        except Exception as e:
            logger.warning(f"[TriggerGuard] 无法获取 AstrBot 主配置，跳过强制接管: {e}")
            return

        ltm_settings = cfg.get("provider_ltm_settings")
        if not isinstance(ltm_settings, dict):
            return
        active_reply = ltm_settings.get("active_reply")
        if not isinstance(active_reply, dict) or not active_reply.get("enable"):
            return

        active_reply["enable"] = False
        try:
            cfg.save_config()
        except Exception as e:
            logger.warning(f"[TriggerGuard] 保存 AstrBot 主配置失败，未能强制接管: {e}")
            return
        logger.warning(
            "[TriggerGuard] 检测到 AstrBot 内置主动回复 "
            "(provider_ltm_settings.active_reply.enable) 和 TriggerGuard 自身的触发判定"
            "同时开启，两边各自独立判定会导致实际触发频率明显偏离你在 TriggerGuard 里"
            "设置的数值。已强制关闭内置主动回复，全部主动回复统一由 TriggerGuard 接管。",
        )

    def _load(self, initial: bool = False) -> None:
        try:
            if not self._data_path.exists():
                self._data_path.parent.mkdir(parents=True, exist_ok=True)
                self._data_path.write_text(
                    json.dumps(_DEFAULT_DATA, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            raw = json.loads(self._data_path.read_text(encoding="utf-8"))
            merged = dict(_DEFAULT_DATA)
            merged.update(raw)

            default_min = self._global_interval_min
            default_max = self._global_interval_max

            # 兼容旧字段名 group_probability/user_probability/user_probability_global。
            group_raw = merged.get("group_interval")
            if group_raw is None:
                group_raw = merged.get("group_probability") or {}
            merged["group_interval"] = _migrate_flat_interval_data(
                {str(k): v for k, v in group_raw.items()}, default_min, default_max,
            )

            user_raw = merged.get("user_interval")
            if user_raw is None:
                user_raw = merged.get("user_probability") or {}
            merged["user_interval"] = _migrate_user_interval_data(
                {str(k): v for k, v in user_raw.items()}, default_min, default_max,
            )

            user_global_raw = merged.get("user_interval_global")
            if user_global_raw is None:
                user_global_raw = merged.get("user_probability_global") or {}
            merged["user_interval_global"] = _migrate_flat_interval_data(
                {str(k): v for k, v in user_global_raw.items()}, default_min, default_max,
            )

            merged.pop("group_probability", None)
            merged.pop("user_probability", None)
            merged.pop("user_probability_global", None)

            merged["blacklist"] = _migrate_entries(
                {str(x) for x in (merged.get("blacklist") or [])},
            )
            merged["blocklist"] = _migrate_entries(
                {str(x) for x in (merged.get("blocklist") or [])},
            )
            merged["active_reply_whitelist"] = _sync_whitelist_with_group_interval(
                merged["group_interval"],
                _migrate_entries(
                    {str(x) for x in (merged.get("active_reply_whitelist") or [])},
                ),
            )
            merged["group_max_length"] = _migrate_flat_int_data(
                {str(k): v for k, v in (merged.get("group_max_length") or {}).items()},
            )
            merged["user_max_length"] = _migrate_user_int_data(
                {str(k): v for k, v in (merged.get("user_max_length") or {}).items()},
            )
            merged["user_max_length_global"] = _migrate_flat_int_data(
                {str(k): v for k, v in (merged.get("user_max_length_global") or {}).items()},
            )
            merged["wake_keywords"] = [
                str(x) for x in (merged.get("wake_keywords") or []) if str(x).strip()
            ]
            self._data = merged
            self._mtime = os.path.getmtime(self._data_path)
        except Exception as e:
            if initial:
                logger.error(f"[TriggerGuard] 加载 overrides.json 失败，使用默认值: {e}")
                self._data = dict(_DEFAULT_DATA)
                self._data["blacklist"] = set()
                self._data["blocklist"] = set()
                self._data["active_reply_whitelist"] = set()
            else:
                logger.warning(f"[TriggerGuard] 重新加载 overrides.json 失败，沿用旧数据: {e}")

    def maybe_reload(self) -> None:
        """若 overrides.json 的修改时间比内存缓存新，则重新加载。供消息/LLM 钩子
        以及 page_api.py 的 UI 读取接口共用。"""
        try:
            mtime = os.path.getmtime(self._data_path)
        except OSError:
            return
        if mtime != self._mtime:
            self._load()

    # ------------------------------------------------------------------ #
    # UI 设置页 (pages/settings) 使用的读写接口，见 page_api.py
    # ------------------------------------------------------------------ #

    def serialize_overrides(self) -> dict[str, Any]:
        """返回可以直接 JSON 序列化、供前端渲染的 overrides 快照。"""
        data = dict(self._data)
        data["blacklist"] = sorted(self._data["blacklist"])
        data["blocklist"] = sorted(self._data["blocklist"])
        data["active_reply_whitelist"] = sorted(self._data["active_reply_whitelist"])
        return data

    def normalize_overrides(self, payload: dict) -> dict[str, Any]:
        """校验并规范化前端提交的 overrides payload，失败抛 ValueError。"""

        def _interval_map(raw: Any, label: str) -> dict[str, dict[str, int]]:
            if raw is None:
                return {}
            if not isinstance(raw, dict):
                raise ValueError(f"{label} 必须是对象")
            out: dict[str, dict[str, int]] = {}
            for k, v in raw.items():
                key = str(k).strip()
                if not key:
                    continue
                if not isinstance(v, dict):
                    raise ValueError(f"{label} 中 {k!r} 的值必须是 {{min,max}} 对象")
                try:
                    lo = int(v.get("min"))
                    hi = int(v.get("max"))
                except (TypeError, ValueError):
                    raise ValueError(f"{label} 中 {k!r} 的 min/max 必须是整数") from None
                if lo < 1:
                    raise ValueError(f"{label} 中 {k!r} 的 min 必须 >= 1")
                if hi < lo:
                    raise ValueError(f"{label} 中 {k!r} 的 max 不能小于 min")
                out[key] = {"min": lo, "max": hi}
            return out

        def _str_list(raw: Any, label: str) -> list[str]:
            if raw is None:
                return []
            if not isinstance(raw, list):
                raise ValueError(f"{label} 必须是数组")
            return [str(x).strip() for x in raw if str(x).strip()]

        def _int_map(raw: Any, label: str) -> dict[str, int]:
            if raw is None:
                return {}
            if not isinstance(raw, dict):
                raise ValueError(f"{label} 必须是对象")
            out: dict[str, int] = {}
            for k, v in raw.items():
                key = str(k).strip()
                if not key:
                    continue
                try:
                    length = int(v)
                except (TypeError, ValueError):
                    raise ValueError(f"{label} 中 {k!r} 的值必须是整数") from None
                if length < 0:
                    raise ValueError(f"{label} 中 {k!r} 的值不能是负数")
                out[key] = length
            return out

        group_interval = _interval_map(payload.get("group_interval"), "group_interval")
        active_reply_whitelist = _sync_whitelist_with_group_interval(
            group_interval,
            set(_str_list(payload.get("active_reply_whitelist"), "active_reply_whitelist")),
        )

        return {
            "_readme": _DEFAULT_DATA["_readme"],
            "group_interval": group_interval,
            "user_interval": _interval_map(payload.get("user_interval"), "user_interval"),
            "user_interval_global": _interval_map(
                payload.get("user_interval_global"), "user_interval_global",
            ),
            "blacklist": set(_str_list(payload.get("blacklist"), "blacklist")),
            "blocklist": set(_str_list(payload.get("blocklist"), "blocklist")),
            "active_reply_whitelist": active_reply_whitelist,
            "group_max_length": _int_map(payload.get("group_max_length"), "group_max_length"),
            "user_max_length": _int_map(payload.get("user_max_length"), "user_max_length"),
            "user_max_length_global": _int_map(
                payload.get("user_max_length_global"), "user_max_length_global",
            ),
            "wake_keywords": _str_list(payload.get("wake_keywords"), "wake_keywords"),
        }

    def save_overrides(self, normalized: dict[str, Any]) -> None:
        """持久化规范化后的 overrides 到磁盘，并立即刷新内存缓存。"""
        serializable = dict(normalized)
        serializable["blacklist"] = sorted(normalized["blacklist"])
        serializable["blocklist"] = sorted(normalized["blocklist"])
        serializable["active_reply_whitelist"] = sorted(normalized["active_reply_whitelist"])
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        self._data_path.write_text(
            json.dumps(serializable, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._data = normalized
        self._mtime = os.path.getmtime(self._data_path)
        # 覆盖数据变了，旧的计数节奏可能不再匹配新的区间，清空让下一条消息重新起算。
        self._counters.clear()
        self._thresholds.clear()

    def _resolve_max_length(self, platform_id: str, group_id: str, sender_id: str) -> int:
        """按 本群用户 > 全局用户(跨群) > 群 > 全局默认 的优先级解析字数上限。

        0 表示不限制。任一层级显式配了 0，就是"这个作用域明确不限制"，不会继续
        往下一层找——用 `in` 判断是否显式配置过，跟"没配置"区分开。
        """
        user_group_key = f"{platform_id}:{group_id}:{sender_id}"
        if user_group_key in self._data["user_max_length"]:
            return self._data["user_max_length"][user_group_key]

        user_global_key = f"{platform_id}:{sender_id}"
        if user_global_key in self._data["user_max_length_global"]:
            return self._data["user_max_length_global"][user_global_key]

        group_key = f"{platform_id}:{group_id}"
        if group_key in self._data["group_max_length"]:
            return self._data["group_max_length"][group_key]

        return self._global_max_length

    def _resolve_scope(
        self, platform_id: str, group_id: str, sender_id: str,
    ) -> tuple[str, int, int]:
        """按 本群用户 > 全局用户(跨群) > 群 > 全局默认 的优先级解析出
        (计数器 key, min, max)。"""
        user_group_key = f"{platform_id}:{group_id}:{sender_id}"
        interval = self._data["user_interval"].get(user_group_key)
        if interval is not None:
            return f"user:{user_group_key}", int(interval["min"]), int(interval["max"])

        user_global_key = f"{platform_id}:{sender_id}"
        interval = self._data["user_interval_global"].get(user_global_key)
        if interval is not None:
            return f"user:{user_global_key}", int(interval["min"]), int(interval["max"])

        group_key = f"{platform_id}:{group_id}"
        interval = self._data["group_interval"].get(group_key)
        if interval is not None:
            return f"group:{group_key}", int(interval["min"]), int(interval["max"])

        return f"group:{group_key}", self._global_interval_min, self._global_interval_max

    def _should_trigger(self, platform_id: str, group_id: str, sender_id: str) -> bool:
        """计数触发判定：每个作用域攒够 [min, max] 之间随机抽到的目标条数普通消息后
        触发一次，然后为下一轮重新抽一个目标值。

        每次都会打一条日志（scope/区间/计数进度），这是排查"配了区间但不触发"最直接
        的证据——尤其能看出实际生效的 scope_key 和区间是不是你以为设置的那个（比如
        覆盖配错了群号/协议实例，实际用的是全局默认区间）。真正触发时始终是 INFO；
        没触发时走 _dlog，默认 DEBUG（安静），`debug_log` 配置打开后升级为 INFO。
        """
        scope_key, lo, hi = self._resolve_scope(platform_id, group_id, sender_id)
        lo = max(1, lo)
        hi = max(lo, hi)

        threshold = self._thresholds.get(scope_key)
        if threshold is None or threshold < lo or threshold > hi:
            threshold = random.randint(lo, hi)
            self._thresholds[scope_key] = threshold

        count = self._counters.get(scope_key, 0) + 1
        triggered = count >= threshold
        message = (
            f"[TriggerGuard] 触发判定 scope={scope_key} 区间=[{lo},{hi}] "
            f"计数={count}/{threshold}{' -> 触发主动回复' if triggered else ''}"
        )
        if triggered:
            logger.info(message)
            self._counters[scope_key] = 0
            self._thresholds[scope_key] = random.randint(lo, hi)
            return True
        self._dlog(message)
        self._counters[scope_key] = count
        return False

    def get_trigger_progress(self, platform_id: str) -> dict[str, Any]:
        """返回某个协议下、已配置的群/用户触发区间目前的计数进度快照，供 UI 展示
        "剩余触发条数 / 预计触发条数"。

        只有该作用域已经处理过至少一条普通消息（因而在 self._thresholds 里已经
        随机抽过一次目标值）才会出现在返回结果里；从没人说过话、或者进程刚重启
        还没积累计数的作用域不会有数据——这是纯内存节奏状态，不是配置，所以没有
        "尚未开始"以外更有意义的默认值可给。
        """

        def _progress(scope_key: str) -> dict[str, int] | None:
            threshold = self._thresholds.get(scope_key)
            if threshold is None:
                return None
            count = self._counters.get(scope_key, 0)
            return {
                "count": count,
                "expected": threshold,
                "remaining": max(0, threshold - count),
            }

        prefix = f"{platform_id}:"

        def _collect(data_key: str, scope_prefix: str) -> dict[str, dict[str, int]]:
            result: dict[str, dict[str, int]] = {}
            for key in self._data[data_key]:
                if not key.startswith(prefix):
                    continue
                entry = _progress(f"{scope_prefix}:{key}")
                if entry is not None:
                    result[key] = entry
            return result

        return {
            "group_progress": _collect("group_interval", "group"),
            "user_progress": _collect("user_interval", "user"),
            "user_global_progress": _collect("user_interval_global", "user"),
        }

    def get_platforms(self) -> list[dict[str, str]]:
        """列出当前已启动的协议实例，供 UI 设置页的侧边协议列表使用。

        webchat 是 AstrBot 自带的网页测试对话，没有"群"的概念，触发控制对它没有
        意义，主动过滤掉。
        """
        platforms: list[dict[str, str]] = []
        manager = getattr(self.context, "platform_manager", None)
        insts = list(getattr(manager, "platform_insts", []) or [])
        for inst in insts:
            try:
                meta = inst.meta()
            except Exception as e:
                logger.warning(f"[TriggerGuard] 读取平台元数据失败: {e}")
                continue
            if meta.name == "webchat":
                continue
            platforms.append(
                {
                    "id": str(meta.id),
                    "name": str(meta.name),
                    "description": str(meta.description or ""),
                },
            )
        platforms.sort(key=lambda p: (p["name"], p["id"]))
        return platforms

    async def get_protocol_stats(self, platform_id: str) -> dict[str, Any]:
        """尝试获取协议实例的统计信息（机器人 ID、加入的群聊数、群成员合计）。

        目前只有 aiocqhttp（OneBot v11）支持自动获取，因为 AstrBot 核心没有跨协议
        统一的"列出群聊"接口——不同协议 SDK 差异太大（Telegram Bot API 甚至不支持
        主动查询机器人加入了哪些群）。其他协议会如实返回 supported=False，而不是
        伪造数据。
        """
        manager = getattr(self.context, "platform_manager", None)
        inst = None
        for candidate in list(getattr(manager, "platform_insts", []) or []):
            try:
                if str(candidate.meta().id) == platform_id:
                    inst = candidate
                    break
            except Exception:
                continue

        if inst is None:
            return {"supported": False, "message": "未找到该协议实例，可能已停止运行。"}

        platform_name = inst.meta().name
        if platform_name != "aiocqhttp":
            return {
                "supported": False,
                "message": (
                    f"协议类型 {platform_name} 暂不支持自动获取群聊列表/统计，"
                    "目前仅 aiocqhttp（OneBot v11）支持，可以手动输入群号。"
                ),
            }

        try:
            client = inst.get_client()
            login_info = await client.call_action("get_login_info")
            raw_groups = await client.call_action("get_group_list")
        except Exception as e:
            logger.warning(f"[TriggerGuard] 获取协议统计失败 ({platform_id}): {e}")
            return {"supported": False, "message": f"调用协议接口失败: {e}"}

        groups = [
            {
                "id": str(g.get("group_id")),
                "name": g.get("group_name") or "",
                "member_count": int(g.get("member_count") or 0),
            }
            for g in (raw_groups or [])
        ]
        groups.sort(key=lambda g: g["id"])

        return {
            "supported": True,
            "bot_id": str(login_info.get("user_id")) if login_info else None,
            "bot_nickname": (login_info or {}).get("nickname"),
            "group_count": len(groups),
            "member_count_sum": sum(g["member_count"] for g in groups),
            "groups": groups,
        }

    async def get_group_members(self, platform_id: str, group_id: str) -> dict[str, Any]:
        """尝试获取某个群的成员列表，供 UI 设置页给 QQ 号输入框做自动补全
        （黑名单/完全屏蔽名单/用户触发区间/用户字数上限这几个面板都要填 QQ 号）。

        跟 get_protocol_stats 一样，目前只有 aiocqhttp（OneBot v11）支持，而且是
        按需/按群拉取，不会在打开设置页时就把机器人所在所有群的成员一次性拉全
        （群成员可能成百上千，一次性拉代价太大）。
        """
        manager = getattr(self.context, "platform_manager", None)
        inst = None
        for candidate in list(getattr(manager, "platform_insts", []) or []):
            try:
                if str(candidate.meta().id) == platform_id:
                    inst = candidate
                    break
            except Exception:
                continue

        if inst is None:
            return {"supported": False, "message": "未找到该协议实例，可能已停止运行。"}

        platform_name = inst.meta().name
        if platform_name != "aiocqhttp":
            return {
                "supported": False,
                "message": (
                    f"协议类型 {platform_name} 暂不支持自动获取群成员列表，"
                    "目前仅 aiocqhttp（OneBot v11）支持，可以手动输入 QQ 号。"
                ),
            }

        try:
            group_id_int = int(group_id)
        except (TypeError, ValueError):
            return {"supported": False, "message": "群号格式不正确。"}

        try:
            client = inst.get_client()
            raw_members = await client.call_action(
                "get_group_member_list", group_id=group_id_int,
            )
        except Exception as e:
            logger.warning(
                f"[TriggerGuard] 获取群成员失败 ({platform_id}:{group_id}): {e}",
            )
            return {"supported": False, "message": f"调用协议接口失败: {e}"}

        members = [
            {
                "id": str(m.get("user_id")),
                "nickname": m.get("nickname") or "",
                "card": m.get("card") or "",
            }
            for m in (raw_members or [])
        ]
        members.sort(key=lambda m: m["id"])

        return {"supported": True, "members": members}

    # ------------------------------------------------------------------ #
    # Handler A: 群消息级别的拦截 / 唤醒判定
    # ------------------------------------------------------------------ #

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL, priority=5000)
    async def guard_message(self, event: AstrMessageEvent) -> None:
        """机器人自身消息、超长消息、拉黑/完全屏蔽名单里的用户，都不会走到
        `_should_trigger`，因此都不计入任何作用域的触发计数器——这是靠下面的
        return 顺序保证的，不是巧合：自身消息第一个挡，然后是字数上限，再是
        blocklist/blacklist，全部都在计数判定之前 return。"""
        if not self._enable:
            return
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        self.maybe_reload()
        self._force_disable_core_active_reply()

        platform_id = str(event.get_platform_id())
        sender_id = str(event.get_sender_id())
        group_id = str(event.get_group_id())
        user_key = f"{platform_id}:{sender_id}"

        if sender_id == str(event.get_self_id()):
            # 机器人自己发的消息不计入计数：某些协议/配置下机器人自身消息也会走一遍
            # AdapterMessageEvent（core 的 ignore_bot_self_message 默认关闭），如果不
            # 挡住，主动回复触发后机器人自己发的那条消息又会被算作一条"普通消息"，
            # 区间设得小（比如 1~1）时会自己跟自己形成无限触发循环。
            self._dlog(
                f"[TriggerGuard] platform={platform_id} group={group_id} "
                "消息发送者是机器人自己，跳过（不计入计数，防止自我触发无限循环）。",
            )
            return

        max_length = self._resolve_max_length(platform_id, group_id, sender_id)
        message_length = len(event.message_str or "")
        if max_length > 0 and message_length > max_length:
            # 超长消息按"完全屏蔽"处理：不进上下文、不触发 LLM（包括直接 @ 唤醒），
            # 防止一条很长的消息把持久化的人设/上下文冲淡或污染。
            logger.info(
                f"[TriggerGuard] 用户 {user_key} 消息长度 {message_length} 超过上限 "
                f"{max_length}，终止事件传播（不记录上下文）。",
            )
            event.should_call_llm(True)
            event.stop_event()
            return

        if user_key in self._data["blocklist"]:
            logger.info(
                f"[TriggerGuard] 用户 {user_key} 命中完全屏蔽名单，终止事件传播（不记录上下文）。",
            )
            event.should_call_llm(True)
            event.stop_event()
            return

        if user_key in self._data["blacklist"]:
            # 保留上下文：不终止事件传播，让 group_chat_context 等 handler 正常记录。
            # 是否触发 LLM 交给 guard_llm_request 在请求前统一拦截。
            logger.info(
                f"[TriggerGuard] 用户 {user_key} 命中拉黑名单，跳过触发判定"
                "（上下文仍会正常记录，真正的 LLM 拦截交给请求前的最终闸口）。",
            )
            return

        if event.is_at_or_wake_command:
            # 已经通过前缀/@/引用唤醒，无需触发判定，交给 AstrBot 默认唤醒链路处理。
            self._dlog(
                f"[TriggerGuard] platform={platform_id} group={group_id} sender={sender_id} "
                "已经是 wake 状态(前缀/@/引用)，跳过触发判定。",
            )
            return

        message_str = event.message_str or ""
        if any(kw in message_str for kw in self._data["wake_keywords"]):
            logger.info(
                f"[TriggerGuard] platform={platform_id} group={group_id} sender={sender_id} "
                "命中唤醒关键词，触发唤醒。",
            )
            event.is_at_or_wake_command = True
            event.is_wake = True
            return

        if not self._active_reply_enable:
            self._dlog(
                f"[TriggerGuard] platform={platform_id} group={group_id} sender={sender_id} "
                "active_reply_enable 未开启，跳过触发判定（这是最常见的“配了区间但从不触发”原因）。",
            )
            return

        group_key = f"{platform_id}:{group_id}"
        if group_key not in self._data["active_reply_whitelist"]:
            self._dlog(
                f"[TriggerGuard] group={group_key} 不在主动回复白名单里，跳过触发判定"
                f"（当前白名单: {sorted(self._data['active_reply_whitelist']) or '空'}）。",
            )
            return

        if self._should_trigger(platform_id, group_id, sender_id):
            event.is_at_or_wake_command = True
            event.is_wake = True

    # ------------------------------------------------------------------ #
    # Handler B: 请求 LLM 前的最终闸口
    # ------------------------------------------------------------------ #

    @filter.on_llm_request(priority=5000)
    async def guard_llm_request(self, event: AstrMessageEvent, req: ProviderRequest) -> None:
        if not self._enable:
            return

        self.maybe_reload()

        user_key = f"{event.get_platform_id()}:{event.get_sender_id()}"
        if user_key in self._data["blocklist"] or user_key in self._data["blacklist"]:
            logger.info(
                f"[TriggerGuard] 用户 {user_key} 被拉黑/屏蔽，已在请求 LLM 前拦截。",
            )
            event.stop_event()

    async def terminate(self) -> None:
        logger.info("[TriggerGuard] 插件已终止")

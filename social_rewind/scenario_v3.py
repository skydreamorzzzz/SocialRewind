"""v3 独立冻结配置；数值仅描述此虚构世界，不改写 v2。"""
from .trade import ContractConfig, InteractionPreferences, TradeScenario, fictional_phone_trade


def strategy_scenario() -> TradeScenario:
    buyer = InteractionPreferences(4100, {
        "1": {"0": 250, "30": 390, "90": 420},
        "3": {"0": 0, "30": 200, "90": 240},
        "7": {"0": -180, "30": 0, "90": 40},
    }, (
        "我的主力手机近期故障，工作通信受到影响，需要可靠的替代设备。",
        "1天交付有明显价值；3天可以接受，但需要合理交换；7天会造成较大不便。",
        "我担心隐藏故障，30天保障非常重要，90天保障的边际价值较低。",
        "我愿意为了合理保障调整时间和价格，也可以退出，退出收益为0。",
        "我不知道卖方的私人效用、成本及可接受条件，只能从自愿披露和行为推断。",
    ))
    seller = InteractionPreferences(3300, {
        "1": {"0": -200, "30": -350, "90": -520},
        "3": {"0": 0, "30": -40, "90": -230},
        "7": {"0": 40, "30": -10, "90": -175},
    }, (
        "我刚买了新手机，希望出售旧设备回收资金，获得较高收益。",
        "我需要迁移个人数据，并希望交付前完成基础检测；1天交付会增加加急和售后风险。",
        "如果有3天准备，就能完成数据迁移和检测，提供30天保障的实际成本会显著降低。",
        "90天保障带来更大的后续责任；我愿意用合同条件交换利益，也可退出，退出收益为0。",
        "我不知道买方私人估值、时间偏好和预算，只能从自愿披露和行为推断。",
    ))
    return TradeScenario(
        scenario_id="fictional-phone-v3-interaction", title="星云 X1：隐藏的交付—保障合作机会",
        fictional=True, product=fictional_phone_trade().product, listed_price=4200,
        buyer_max_price=4100, seller_min_price=3300, contract=ContractConfig(buyer, seller),
    )

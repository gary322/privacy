// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

// UVCC_REQ_GROUP: uvcc_group_469c18d206813c25

import {IERC20} from "openzeppelin-contracts/token/ERC20/IERC20.sol";
import {SafeERC20} from "openzeppelin-contracts/token/ERC20/utils/SafeERC20.sol";

/// @notice Provider bond registry (v1). Providers escrow AVL bonds and can be slashed.
contract ProviderBondRegistry {
    using SafeERC20 for IERC20;

    IERC20 public immutable avl;
    address public jobLedger;
    address public immutable owner;

    mapping(address => uint256) public bondOf;

    event BondDeposited(address indexed provider, uint256 amount);
    event BondWithdrawn(address indexed provider, uint256 amount);
    event BondSlashed(address indexed provider, address indexed to, uint256 amount);

    modifier onlyLedger() {
        require(jobLedger != address(0) && msg.sender == jobLedger, "only ledger");
        _;
    }

    constructor(IERC20 avl_, address jobLedger_) {
        avl = avl_;
        owner = msg.sender;
        jobLedger = jobLedger_;
    }

    function setJobLedger(address jobLedger_) external {
        require(msg.sender == owner, "only owner");
        require(jobLedger == address(0), "already set");
        require(jobLedger_ != address(0), "ledger=0");
        jobLedger = jobLedger_;
    }

    function depositBond(uint256 amount) external {
        require(amount > 0, "amount=0");
        bondOf[msg.sender] += amount;
        avl.safeTransferFrom(msg.sender, address(this), amount);
        emit BondDeposited(msg.sender, amount);
    }

    function withdrawBond(uint256 amount) external {
        require(amount > 0, "amount=0");
        uint256 bal = bondOf[msg.sender];
        require(bal >= amount, "insufficient");
        bondOf[msg.sender] = bal - amount;
        avl.safeTransfer(msg.sender, amount);
        emit BondWithdrawn(msg.sender, amount);
    }

    function slash(address provider, address to, uint256 amount) external onlyLedger {
        require(amount > 0, "amount=0");
        uint256 bal = bondOf[provider];
        require(bal >= amount, "insufficient");
        bondOf[provider] = bal - amount;
        avl.safeTransfer(to, amount);
        emit BondSlashed(provider, to, amount);
    }
}



// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

// UVCC_REQ_GROUP: uvcc_group_469c18d206813c25,uvcc_group_014ea1c86d0fe430

import {ECDSA} from "openzeppelin-contracts/utils/cryptography/ECDSA.sol";
import {EIP712} from "openzeppelin-contracts/utils/cryptography/EIP712.sol";

import {AVLStakingManager} from "./AVLStakingManager.sol";
import {ProviderBondRegistry} from "./ProviderBondRegistry.sol";

/// @notice UVCC job ledger (v1): records jobs and verifies final proof commits + signatures.
contract UVCCJobLedger is EIP712 {
    using ECDSA for bytes32;
    // Canonical window constants (privacy_new.txt §8.2).
    uint256 public constant CHALLENGE_WINDOW = 3 days; // 259200
    uint256 public constant RESPONSE_WINDOW = 1 days; // 86400

    // Domain strings with NUL terminators (wire-exact).
    bytes internal constant DS_JOBID = hex"555643432e6a6f6269642e763100"; // "UVCC.jobid.v1\0"

    // EIP-712 type hashes (privacy_new.txt §3.2).
    bytes32 public constant POLICY_COMMIT_TYPEHASH =
        keccak256(
            "PolicyCommit(bytes32 jobId,bytes32 policyHash,bytes32 sidHash,bytes32 sgirHash,bytes32 runtimeHash,bytes32 fssDirHash,bytes32 preprocHash,uint8 backend,uint64 epoch)"
        );
    bytes32 public constant FINAL_COMMIT_TYPEHASH = keccak256("FinalCommit(bytes32 jobId,bytes32 policyHash,bytes32 finalRoot,bytes32 resultHash)");

    enum Backend {
        CRYPTO_CC_3PC,
        GPU_TEE
    }

    struct PolicyCommit {
        bytes32 jobId;
        bytes32 policyHash;
        bytes32 sidHash;
        bytes32 sgirHash;
        bytes32 runtimeHash;
        bytes32 fssDirHash;
        bytes32 preprocHash;
        uint8 backend;
        uint64 epoch;
    }

    AVLStakingManager public immutable staking;
    ProviderBondRegistry public immutable bonds;

    struct Job {
        bytes32 policyHash;
        bytes32 sidHash;
        bytes32 sgirHash;
        bytes32 runtimeHash;
        bytes32 fssDirHash;
        bytes32 preprocHash;
        Backend backend;
        uint64 epoch;
        address creator;
        address[3] parties;
        uint64 createdAt;
        bool finalized;
        bytes32 finalRoot;
        bytes32 resultHash;
        uint64 finalizedAt;
        bool challenged;
        address challenger;
        uint64 challengedAt;
        bool resolved;
        bool slashed;
    }

    mapping(bytes32 => Job) public jobs;

    event JobCreated(bytes32 indexed jobId, bytes32 indexed policyHash, address indexed creator, address[3] parties);
    event FinalSubmitted(bytes32 indexed jobId, bytes32 finalRoot, bytes32 resultHash);
    event Challenged(bytes32 indexed jobId, address indexed challenger);
    event Resolved(bytes32 indexed jobId, bool slashed);

    constructor(AVLStakingManager staking_, ProviderBondRegistry bonds_) EIP712("UVCC", "1") {
        staking = staking_;
        bonds = bonds_;
    }

    function computeJobId(bytes32 policyHash, bytes32 clientNonce32) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(DS_JOBID, policyHash, clientNonce32));
    }

    function hashPolicyCommit(PolicyCommit calldata pc) public view returns (bytes32) {
        bytes32 structHash = keccak256(
            abi.encode(
                POLICY_COMMIT_TYPEHASH,
                pc.jobId,
                pc.policyHash,
                pc.sidHash,
                pc.sgirHash,
                pc.runtimeHash,
                pc.fssDirHash,
                pc.preprocHash,
                pc.backend,
                pc.epoch
            )
        );
        return _hashTypedDataV4(structHash);
    }

    function hashFinalCommit(bytes32 jobId, bytes32 policyHash, bytes32 finalRoot, bytes32 resultHash) public view returns (bytes32) {
        bytes32 structHash = keccak256(abi.encode(FINAL_COMMIT_TYPEHASH, jobId, policyHash, finalRoot, resultHash));
        return _hashTypedDataV4(structHash);
    }

    function _recoverSigner(bytes32 digest, bytes calldata sig65) internal pure returns (address) {
        // Accept v in {0,1} or {27,28}, normalize to {27,28}, then delegate to OZ ECDSA.
        require(sig65.length == 65, "bad sig len");
        bytes32 r;
        bytes32 s;
        uint8 v;
        assembly {
            r := calldataload(sig65.offset)
            s := calldataload(add(sig65.offset, 32))
            v := byte(0, calldataload(add(sig65.offset, 64)))
        }
        if (v < 27) v += 27;
        bytes memory sig = abi.encodePacked(r, s, v);
        return ECDSA.recover(digest, sig);
    }

    function createJob(PolicyCommit calldata pc, address[3] calldata partySigners, bytes calldata sig0, bytes calldata sig1, bytes calldata sig2)
        external
        returns (bytes32 jobId)
    {
        require(pc.backend <= uint8(Backend.GPU_TEE), "bad backend");
        jobId = pc.jobId;
        Job storage j = jobs[jobId];
        require(j.createdAt == 0, "job exists");
        bytes32 digest = hashPolicyCommit(pc);
        require(_recoverSigner(digest, sig0) == partySigners[0], "sig0 bad");
        require(_recoverSigner(digest, sig1) == partySigners[1], "sig1 bad");
        require(_recoverSigner(digest, sig2) == partySigners[2], "sig2 bad");

        j.policyHash = pc.policyHash;
        j.sidHash = pc.sidHash;
        j.sgirHash = pc.sgirHash;
        j.runtimeHash = pc.runtimeHash;
        j.fssDirHash = pc.fssDirHash;
        j.preprocHash = pc.preprocHash;
        j.backend = Backend(pc.backend);
        j.epoch = pc.epoch;
        j.creator = msg.sender;
        j.parties = partySigners;
        j.createdAt = uint64(block.timestamp);
        emit JobCreated(jobId, j.policyHash, msg.sender, partySigners);
    }

    function getParties(bytes32 jobId) external view returns (address[3] memory) {
        Job storage j = jobs[jobId];
        require(j.createdAt != 0, "missing job");
        return j.parties;
    }

    function submitFinal(
        bytes32 jobId,
        bytes32 finalRoot,
        bytes32 resultHash,
        bytes calldata sigP0,
        bytes calldata sigP1,
        bytes calldata sigP2
    ) external {
        Job storage j = jobs[jobId];
        require(j.createdAt != 0, "missing job");
        require(!j.finalized, "already finalized");

        bytes32 digest = hashFinalCommit(jobId, j.policyHash, finalRoot, resultHash);
        require(_recoverSigner(digest, sigP0) == j.parties[0], "sig0 bad");
        require(_recoverSigner(digest, sigP1) == j.parties[1], "sig1 bad");
        require(_recoverSigner(digest, sigP2) == j.parties[2], "sig2 bad");

        j.finalized = true;
        j.finalRoot = finalRoot;
        j.resultHash = resultHash;
        j.finalizedAt = uint64(block.timestamp);
        emit FinalSubmitted(jobId, finalRoot, resultHash);
    }

    function challenge(bytes32 jobId) external {
        Job storage j = jobs[jobId];
        require(j.finalized, "not finalized");
        require(!j.challenged, "already challenged");
        require(block.timestamp <= uint256(j.finalizedAt) + CHALLENGE_WINDOW, "challenge window passed");
        j.challenged = true;
        j.challenger = msg.sender;
        j.challengedAt = uint64(block.timestamp);
        emit Challenged(jobId, msg.sender);
    }

    function resolve(bytes32 jobId, bool slash) external {
        Job storage j = jobs[jobId];
        require(j.challenged, "not challenged");
        require(!j.resolved, "already resolved");
        require(block.timestamp > uint256(j.challengedAt) + RESPONSE_WINDOW, "response window not passed");
        j.resolved = true;
        j.slashed = slash;
        // v1 demo: slashing transfer is left to a governance/external adjudication path.
        emit Resolved(jobId, slash);
    }

    // EIP-712 signature validation uses OpenZeppelin's ECDSA.recover and EIP712._hashTypedDataV4.
}



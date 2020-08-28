// ESTA-Community-Helper for Python (https://issues.sbb.ch/browse/ESTA-3822): Status alpha!
// https://code.sbb.ch/projects/KD_ESTA_BLUEPRINTS/repos/esta-python-helper/browse
// https://code.sbb.ch/projects/KD_ESTA_BLUEPRINTS/repos/esta-python-lib/browse/Jenkinsfile
library identifier: 'python-helper@master',
        retriever: modernSCM(
                [$class       : 'GitSCMSource',
                 credentialsId: 'fsosebuild',
                 remote       : 'ssh://git@code.sbb.ch:7999/kd_esta_blueprints/esta-python-helper.git'])


pipeline {
    // aws label required, no access to internet from default vias nodes: https://issues.sbb.ch/servicedesk/customer/portal/1/CLEW-895
    agent { label 'aws' }
    options {
        timestamps()
        disableConcurrentBuilds()
        // https://stackoverflow.com/questions/39542485/how-to-write-pipeline-to-discard-old-builds
        // maximal 5 Builds und maximal 30 Tage
        buildDiscarder(logRotator(numToKeepStr: '5', artifactNumToKeepStr: '5', daysToKeepStr: '30', artifactDaysToKeepStr: '30'))
    }
    environment {
        // access token with repo:status permission created via https://github.com/settings/tokens in personal account
        // uploaded to Jenkins via https://ssp.app.ose.sbb-cloud.net/wzu/jenkinscredentials
        // list of credentials: https://ci.sbb.ch/job/KS_PFI/credentials/
        GITHUB_TOKEN = credentials('19fbbc7a-7243-431c-85d8-0f1cc63d413b')

        //-------------------------------------------------------------
        // Configuration for base image
        //-------------------------------------------------------------
        // Enter the name of your Artifactory Docker Repository.
        //   Artifactory Docker Repositories can be created on:
        //   https://ssp.app.ose.sbb-cloud.net --> WZU-Dienste --> Artifactory
        ARTIFACTORY_PROJECT = 'pfi'
        BASE_IMAGE_NAME = 'rsp-workspace'

        //-------------------------------------------------------------
        // Configuration for base image deployment
        //-------------------------------------------------------------
        // https://code.sbb.ch/projects/KD_ESTA/repos/pipeline-helper/browse/src/ch/sbb/util/OcClusters.groovy
        OPENSHIFT_CLUSTER = "otc_prod_gpu"
        OPENSHIFT_CLUSTER_URL = "https://master.gpu.otc.sbb.ch:8443"
        OPENSHIFT_PROJECT = "pfi-digitaltwin-ci"
        HELM_CHART = 'rsp_workspace'
        // https://ssp.app.ose.sbb-cloud.net/ose/newserviceaccount
        // https://ci.sbb.ch/job/KS_PFI/credentials/
        SERVICE_ACCOUNT_TOKEN = 'aaff533e-7ebe-469d-a13a-31f786245d1b'
    }
    stages {
        stage('github pending') {
            steps {
                // https://developer.github.com/v3/repos/statuses/
                script {
                    sh """
echo $GITHUB_TOKEN > token.txt
curl --insecure -v --request POST -H "Authorization: token ${
                        GITHUB_TOKEN
                    }" https://api.github.com/repos/SchweizerischeBundesbahnen/rsp/statuses/${
                        GIT_COMMIT
                    } --data '{ "state": "pending", "target_url": "'${BUILD_URL}'", "description": "The build has started!", "context": "continuous-integration/jenkins" }'
"""
                }
            }
        }
        stage('init git submodules') {
            steps {
                script {
                    sh """
git submodule update --init --recursive
"""
                }
            }
        }
        // if we're on master, tag the docker image with the new semantic version
        stage("Build and Tag Docker Image if on master") {
            when {
                allOf {
                    expression { BRANCH_NAME == 'master' }
                }
            }
            steps {
                script {
                    echo """cloud_buildDockerImage()"""
                    echo """GIT_COMMIT=${env.GIT_COMMIT}"""
                    cloud_buildDockerImage(
                            artifactoryProject: env.ARTIFACTORY_PROJECT,
                            ocApp: env.BASE_IMAGE_NAME,
                            ocAppVersion: env.GIT_COMMIT,
                            // we must be able to access rsp_environment.yml from within docker root !
                            // https://confluence.sbb.ch/display/CLEW/Pipeline+Helper#PipelineHelper-cloud_buildDockerImage()-BuildfromownDockerfile
                            dockerDir: '.',
                            dockerfilePath: 'docker/Dockerfile'
                    )
                    cloud_tagDockerImage(
                            artifactoryProject: env.ARTIFACTORY_PROJECT,
                            ocApp: env.BASE_IMAGE_NAME,
                            tag: env.GIT_COMMIT,
                            targetTag: "latest"
                    )
                }
            }
        }
        stage("Integration Test Notebooks") {
            when {
                anyOf {
                    // skip on pr: https://jenkins.io/doc/book/pipeline/multibranch/
                    expression { env.CHANGE_ID == null }
                }
            }
            steps {
                script {
                    cloud_helmchartsDeploy(
                            cluster: OPENSHIFT_CLUSTER,
                            project: env.OPENSHIFT_PROJECT,
                            credentialId: SERVICE_ACCOUNT_TOKEN,
                            chart: env.HELM_CHART,
                            release: 'rsp-ci-' + GIT_COMMIT,
                            additionalValues: [
                                    // TODO the docker image should be extracted from this repo since they have independent lifecycles!
                                    RspWorkspaceVersion: "latest",
                                    RspVersion         : GIT_COMMIT
                            ]
                    )
                    echo "Logs can be found under https://master.gpu.otc.sbb.ch:8443/console/project/pfi-digitaltwin-ci/browse/pods/rsp-ci-$GIT_COMMIT-test-pod?tab=logs"
                    cloud_helmchartsTest(
                            cluster: OPENSHIFT_CLUSTER,
                            project: env.OPENSHIFT_PROJECT,
                            credentialId: SERVICE_ACCOUNT_TOKEN,
                            release: 'rsp-ci-$GIT_COMMIT',
                            timeoutInSeconds: 2700
                    )
                    echo "helm test succesful."
                }
            }
        }
    }
    post {
        failure {
            // https://developer.github.com/v3/repos/statuses/
            sh """
curl --insecure -v --request POST -H "Authorization: token ${
                GITHUB_TOKEN
            }" https://api.github.com/repos/SchweizerischeBundesbahnen/rsp/statuses/${
                GIT_COMMIT
            } --data '{ "state": "failure", "target_url": "'${BUILD_URL}'", "description": "The build has failed!", "context": "continuous-integration/jenkins" }'
"""
        }
        success {
            // https://developer.github.com/v3/repos/statuses/
            sh """
curl --insecure -v --request POST -H "Authorization: token ${
                GITHUB_TOKEN
            }" https://api.github.com/repos/SchweizerischeBundesbahnen/rsp/statuses/${
                GIT_COMMIT
            } --data '{ "state": "success", "target_url": "'${BUILD_URL}'", "description": "The build has succeeded!", "context": "continuous-integration/jenkins" }'
"""
        }
        always {
            withCredentials([string(credentialsId: SERVICE_ACCOUNT_TOKEN, variable: 'TOKEN')]) {
                echo """get logs from https://master.gpu.otc.sbb.ch:8443/console/project/pfi-digitaltwin-ci/browse/pods/rsp-ci-$GIT_COMMIT-test-pod?tab=logs"""
                sh """
oc login $OPENSHIFT_CLUSTER_URL --token=$TOKEN --insecure-skip-tls-verify=true
oc project $OPENSHIFT_PROJECT

oc logs rsp-ci-$GIT_COMMIT-test-pod || true

oc delete rsp-ci-$GIT_COMMIT-test-pod || true
helm delete rsp-ci-$GIT_COMMIT || true

# delete all failed test pods older than 1 day (https://stackoverflow.com/questions/48934491/kubernetes-how-to-delete-pods-based-on-age-creation-time/48960060#48960060)
oc get pods --field-selector status.phase=Failed -o go-template --template '{{range .items}}{{.metadata.name}} {{.metadata.creationTimestamp}}{{"\\n"}}{{end}}' | awk '\$2 <= "'\$(date -d 'yesterday' -Ins --utc | sed 's/+0000/Z/')'" { print \$1 }' | fgrep test-pod | xargs --no-run-if-empty oc delete pod
"""
            }
            archiveArtifacts artifacts: 'rsp_*.png', onlyIfSuccessful: true, allowEmptyArchive: true
            cleanWs()
        }
    }
}
